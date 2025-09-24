import os
import json
import requests
import certifi
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from scipy.optimize import curve_fit

# --- Setup cache and plots directories ---
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Fetch + cache function ---
def fetch_and_cache(url, fname):
    path = os.path.join(CACHE_DIR, fname)

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
        
    response = requests.get(url, verify=certifi.where(), timeout=30)
    response.raise_for_status()
    data = response.json()

    with open(path, "w") as f:
        json.dump(data, f)

    return data

# --- User parameters ---
COUNTRY = "Italy"
YEAR = 2024
SESSION_TYPE = "Practice"
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"] 
SECONDS_SAVED_PER_LAP_FUEL = 0.045

# --- Get sessions ---
sessions_url = f"https://api.openf1.org/v1/sessions?country_name={COUNTRY}&session_type={SESSION_TYPE}&year={YEAR}"
sessions = fetch_and_cache(sessions_url, f"sessions_{COUNTRY}_{SESSION_TYPE}_{YEAR}.json")
practice_session_keys = [s["session_key"] for s in sessions[:3]]
print("Practice sessions:", practice_session_keys)

# --- Store data for combined plot ---
combined_fits = []

# --- Loop over compounds ---
for COMPOUND in COMPOUNDS:
    rows = []
    for session in practice_session_keys:
        stints = fetch_and_cache(f"https://api.openf1.org/v1/stints?session_key={session}", f"stints_{session}.json")
        laps = fetch_and_cache(f"https://api.openf1.org/v1/laps?session_key={session}", f"laps_{session}.json")
        stints = [stint for stint in stints if (stint["lap_end"] - stint["lap_start"]) > 5]

        # Group laps by driver
        laps_by_driver = defaultdict(dict)
        for lap in laps:
            dnum = lap.get("driver_number")
            ln = lap.get("lap_number")
            if dnum is not None and ln is not None:
                laps_by_driver[dnum][ln] = lap

        # Iterate stints
        for stint in stints:
            tyre = stint.get("compound")
            if not tyre or tyre.upper() != COMPOUND.upper():
                continue
            dnum = stint.get("driver_number")
            try:
                start = int(stint.get("lap_start"))
                end = int(stint.get("lap_end"))
            except (TypeError, ValueError):
                continue
            stint_length = max(0, end - start)
            tyre_age_start = int(stint.get("tyre_age_at_start", 0))

            driver_laps = laps_by_driver.get(dnum, {})
            for ln in range(start, end):
                lap = driver_laps.get(ln)
                if not lap or lap.get("is_pit_out_lap"):
                    continue
                try:
                    lap_time = (float(lap["duration_sector_1"])
                              + float(lap["duration_sector_2"])
                              + float(lap["duration_sector_3"]))
                except (TypeError, KeyError, ValueError):
                    continue

                tyre_age = tyre_age_start + (ln - start)
                rows.append({
                    "lap_time": lap_time,
                    "tyre_age": tyre_age,
                    "driver": dnum,
                    "session": session,
                    "lap_number": ln,
                    "stint_start": start,
                    "stint_end": end,
                    "stint_length": stint_length,
                    "tyre_age_at_start": tyre_age_start,
                    "stint_number": stint.get("stint_number")
                })

    # --- Push lap filter + drop first lap of each stint ---
    rows_filtered = []
    groups = defaultdict(list)
    for r in rows:
        key = (r["driver"], r["session"], r["stint_number"])
        groups[key].append(r)

    for group in groups.values():
        lap_times = np.array([r["lap_time"] for r in group])
        median_time = np.median(lap_times)
        for r in group:
            if r["lap_number"] == r["stint_start"]:
                continue  # drop first lap of stint
            if r["lap_time"] > median_time - 1.5:  # drop push laps
                rows_filtered.append(r)

    rows = rows_filtered

    # --- Fuel correction ---
    detailed = []
    for r in rows:
        ln = r["lap_number"]
        start = r["stint_start"]
        laps_done = ln - start
        stint_len = r["stint_length"]
        start_fuel_laps = stint_len + 2
        remaining_fuel_laps = max(0, start_fuel_laps - laps_done)

        penalty_sec = remaining_fuel_laps * SECONDS_SAVED_PER_LAP_FUEL
        fuel_corrected_time_zero = r["lap_time"] - penalty_sec

        newrow = r.copy()
        newrow.update({
            "fuel_corrected_time_zero": fuel_corrected_time_zero,
        })
        detailed.append(newrow)

    # --- Sequential anomaly filter ---
    rows_filtered_seq = []
    last_mean = None
    THRESHOLD = 1.03  # Remove if > 1.03 × last accepted lap

    sorted_indices = np.argsort([r["tyre_age"] for r in detailed])
    for idx in sorted_indices:
        r = detailed[idx]
        lap_time = r["fuel_corrected_time_zero"]
        if last_mean is None or lap_time <= THRESHOLD * last_mean:
            rows_filtered_seq.append(r)
            last_mean = lap_time

    detailed = rows_filtered_seq

    # --- Compute mean per tyre age ---
    ages_clean = np.array([r["tyre_age"] for r in detailed])
    fuel_zero_clean = np.array([r["fuel_corrected_time_zero"] for r in detailed])
    unique_ages = np.array(sorted(set(ages_clean)))
    mean_times_fuel_zero = np.array([fuel_zero_clean[ages_clean == a].mean() for a in unique_ages])

    # --- Mask very low tyre ages ---
    mask = unique_ages >= 2
    x = unique_ages[mask]
    y = mean_times_fuel_zero[mask]

    # --- Avoid log(0) for exponential fits ---
    y_safe = np.clip(y, 1e-6, None)
    log_y = np.log(y_safe)
    X = x.reshape(-1, 1)

    # # --- 1. RANSAC exponential fit ---
    # ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=0.5,
    #                          residual_threshold=0.05, random_state=42)
    # ransac.fit(X, log_y)
    # b_r = ransac.estimator_.coef_[0]
    # a_r = np.exp(ransac.estimator_.intercept_)
    x_plot = np.linspace(x.min(), x.max(), 200)
    # y_fit_r = a_r * np.exp(b_r * x_plot)

    # # --- 2. Huber exponential fit ---
    # huber = HuberRegressor()
    # huber.fit(X, log_y)
    # b_h = huber.coef_[0]
    # a_h = np.exp(huber.intercept_)
    # y_fit_h = a_h * np.exp(b_h * x_plot)

    # --- 3. Offset exponential fit ---
    def exp_offset_full(x, a, b, c):
        return c + a * np.exp(b * x)

    p0 = (1, 0.05, np.min(y))
    try:
        popt, _ = curve_fit(exp_offset_full, x, y, p0=p0, maxfev=5000)
        a_fit, b_fit, c_fit = popt
    except RuntimeError:
        a_fit, b_fit, c_fit = np.nan, np.nan, np.nan

    y_fit_offset = exp_offset_full(x_plot, a_fit, b_fit, c_fit)

    # --- Plot per compound ---
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, s=10, alpha=0.5, label="Fuel-corrected mean")
    plt.plot(x_plot, y_fit_offset, "m-", linewidth=2,
             label=f"Offset Exp: y={c_fit:.3f}+{a_fit:.3f}·exp({b_fit:.4f}x)")
    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Lap time (s)")
    plt.title(f"Exponential Fit for {COMPOUND}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_fname = os.path.join(PLOTS_DIR, f"exp_fit_offset_{COMPOUND}.png")
    plt.savefig(plot_fname, dpi=300)
    plt.close()
    print(f"Saved offset exponential fit plot for {COMPOUND} to {plot_fname}")

    combined_fits.append((COMPOUND, (a_fit, b_fit, c_fit)))

# --- Combined plot for all compounds ---
plt.figure(figsize=(10,6))
colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "grey"}
x_plot = np.linspace(0, 30, 300)
for compound, params in combined_fits:
    a_fit, b_fit, c_fit = params
    y_plot = exp_offset_full(x_plot, a_fit, b_fit, c_fit)
    plt.plot(x_plot, y_plot, color=colors.get(compound, "black"), linewidth=2,
             label=f"{compound}: y={c_fit:.2f}+{a_fit:.2f}·exp({b_fit:.4f}x)")

plt.xlabel("Tyre age (laps)")
plt.ylabel("Lap time (s)")
plt.title("Offset Exponential Fits for All Compounds")
plt.grid(True, alpha=0.3)
plt.legend()
combined_plot_fname = os.path.join(PLOTS_DIR, "exp_fit_offset_all_compounds.png")
plt.savefig(combined_plot_fname, dpi=300)
plt.close()
print(f"Saved combined offset exponential fit plot to {combined_plot_fname}")
