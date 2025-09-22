import os
import json
import requests
import certifi
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
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
COUNTRY = "Hungary"
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
        stints = list(filter(lambda stint: (stint["lap_end"] - stint["lap_start"]) > 5, stints))

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
        lap_times = [r["lap_time"] for r in group]
        median_time = np.median(lap_times)
        for r in group:
            if r["lap_number"] == r["stint_start"]:
                continue  # drop first lap of stint
            if r["lap_time"] > median_time - 1.5:  # drop push laps
                rows_filtered.append(r)

    rows = rows_filtered

    # --- Fuel correction ---
    detailed = []
    starting_fuel_laps_list = []
    for r in rows:
        stint_len = r["stint_length"]
        start_fuel_laps = stint_len + 2
        starting_fuel_laps_list.append(start_fuel_laps)

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
            "start_fuel_laps": start_fuel_laps,
            "remaining_fuel_laps": remaining_fuel_laps,
            "penalty_seconds": penalty_sec,
            "fuel_corrected_time_zero": fuel_corrected_time_zero,
        })
        detailed.append(newrow)

    # --- Filter anomalies ---
    lap_times = np.array([r["lap_time"] for r in detailed])
    ages = np.array([r["tyre_age"] for r in detailed])
    fuel_zero = np.array([r["fuel_corrected_time_zero"] for r in detailed])
    ANOMALY_THRESHOLD = min(lap_times) * 1.15
    mask = (lap_times <= ANOMALY_THRESHOLD) & np.isfinite(lap_times)
    lap_times_clean = lap_times[mask]
    ages_clean = ages[mask]
    fuel_zero_clean = fuel_zero[mask]

    print(f"{COMPOUND}: After filtering lap_time <= {ANOMALY_THRESHOLD:.3f}: {len(lap_times_clean)} rows remain.")

    # --- Calculate mean per tyre age ---
    unique_ages = np.array(sorted(set(ages_clean)))
    mean_times_fuel_zero = np.array([fuel_zero_clean[ages_clean == a].mean() for a in unique_ages])
    counts_per_age = np.array([np.sum(ages_clean == a) for a in unique_ages])

    # --- Weighted exponential fit ---
    def exp_model(x, a, b):
        return a * np.exp(b * x)

    weights = counts_per_age / counts_per_age.max()
    popt, _ = curve_fit(exp_model, unique_ages, mean_times_fuel_zero,
                        p0=(mean_times_fuel_zero[0], 0.01), sigma=1/weights)
    a_fit, b_fit = popt

    x_smooth = np.linspace(unique_ages.min(), unique_ages.max(), 200)
    y_fit = exp_model(x_smooth, a_fit, b_fit)

    # Save per-compound fit for combined plot
    combined_fits.append((COMPOUND, x_smooth, y_fit))

    # --- Individual compound plot ---
    plt.figure(figsize=(10,6))
    plt.scatter(ages_clean, fuel_zero_clean, alpha=0.4, s=10, label="Fuel-corrected laps")
    plt.plot(unique_ages, mean_times_fuel_zero, linewidth=2, label="Mean per tyre age")
    plt.plot(x_smooth, y_fit, color="red", linewidth=2,
             label=f"Exp fit: y = {a_fit:.3f}·e^({b_fit:.3f}x)")

    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Lap time (s)")
    plt.title(f"{COMPOUND} Lap Times vs Tyre Age (Fuel-corrected, Zero)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{COMPOUND}_exp_fit_fuel_zero.png"), dpi=300)
    plt.close()

    print(f"Saved exponential fit plot for {COMPOUND}")

# --- Combined overlay plot ---
plt.figure(figsize=(10,6))
for compound, x_smooth, y_fit in combined_fits:
    plt.plot(x_smooth, y_fit, linewidth=2, label=compound)

plt.xlabel("Tyre age (laps)")
plt.ylabel("Lap time (s)")
plt.title("Tyre Degradation Comparison (Fuel-corrected, Zero)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "Combined_Compounds_Comparison.png"), dpi=300)
plt.close()

print("Saved combined compound comparison plot.")
