import os
import json
import time
import requests
import certifi
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Potentially add a sleep
def fetch_and_cache(url, fname):
    path = os.path.join(CACHE_DIR, fname)

    # If we already have a cached copy, use it
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
        
    # Otherwise, fetch from the API
    response = requests.get(url, verify=certifi.where(), timeout=30)
    response.raise_for_status()
    data = response.json()

    with open(path, "w") as f:
        json.dump(data, f)

    return data

# USER PARAMETERS
COUNTRY = "Hungary"
YEAR = 2024
SESSION_TYPE = "Practice"
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"] 
ANOMALY_THRESHOLD = 74.0
FUEL_PER_LAP = 1.5
SECONDS_SAVED_PER_LAP_FUEL = 0.045

# --- get sessions ---
sessions_url = f"https://api.openf1.org/v1/sessions?country_name={COUNTRY}&session_type={SESSION_TYPE}&year={YEAR}"
sessions = fetch_and_cache(sessions_url, f"sessions_{COUNTRY}_{SESSION_TYPE}_{YEAR}.json")
practice_session_keys = [s["session_key"] for s in sessions[:3]]
print(practice_session_keys)

for COMPOUND in COMPOUNDS:
    rows = []
    for session in practice_session_keys:
        stints = fetch_and_cache(f"https://api.openf1.org/v1/stints?session_key={session}", f"stints_{session}.json")
        laps = fetch_and_cache(f"https://api.openf1.org/v1/laps?session_key={session}", f"laps_{session}.json")
        print(stints[0])

        # store each lap
        laps_by_driver = defaultdict(dict)
        for lap in laps:
            dnum = lap.get("driver_number")
            ln = lap.get("lap_number")
            if dnum is not None and ln is not None:
                laps_by_driver[dnum][ln] = lap

        # iterate stints
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
                    "tyre_age_at_start": tyre_age_start
                })

    # --- fuel correction ---
    detailed = []
    starting_fuel_laps_list = []
    for r in rows:
        stint_len = r["stint_length"]
        start_fuel_laps = stint_len + 2
        starting_fuel_laps_list.append(start_fuel_laps)

    baseline_start_fuel = int(np.median(starting_fuel_laps_list)) if starting_fuel_laps_list else 0

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

    # --- filter anomalies ---
    lap_times = np.array([r["lap_time"] for r in detailed])
    ages = np.array([r["tyre_age"] for r in detailed])
    fuel_zero = np.array([r["fuel_corrected_time_zero"] for r in detailed])
    ANOMALY_THRESHOLD = min(lap_times)*1.15
    mask = (lap_times <= ANOMALY_THRESHOLD) & np.isfinite(lap_times)
    lap_times_clean = lap_times[mask]
    ages_clean = ages[mask]
    fuel_zero_clean = fuel_zero[mask]

    print(f"  After filtering lap_time <= {ANOMALY_THRESHOLD}: {len(lap_times_clean)} rows remain.")

    # --- plotting ---
    unique_ages = sorted(set(ages_clean))
    mean_times_raw = [lap_times_clean[ages_clean == a].mean() for a in unique_ages]
    mean_times_fuel_zero = [fuel_zero_clean[ages_clean == a].mean() for a in unique_ages]

    # raw + mean
    plt.figure(figsize=(10,6))
    plt.scatter(ages_clean, lap_times_clean, alpha=0.4, s=10, label="Lap times")
    plt.plot(unique_ages, mean_times_raw, linewidth=2, label="Mean per tyre age")
    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Lap time (s)")
    plt.title(f"{COMPOUND} Lap Times vs Tyre Age (Mean)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{COMPOUND}_scatter_mean_raw.png"), dpi=300)
    plt.close()

    # fuel corrected (zero baseline)

    # Data
    x = np.array(unique_ages)
    y = np.array(mean_times_fuel_zero)

    # Avoid log(0) errors
    y_safe = np.where(y <= 0, 1e-6, y)

    # Log-linear fit
    log_y = np.log(y_safe)
    coeffs = np.polyfit(x, log_y, 1)   # linear fit: log(y) = b*x + log(a)
    b = coeffs[0]
    log_a = coeffs[1]
    a = np.exp(log_a)

    # Generate smooth curve
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_fit = a * np.exp(b * x_smooth)

    # Plot
    plt.figure(figsize=(10,6))
    plt.scatter(ages_clean, fuel_zero_clean, alpha=0.4, s=10, label="Fuel-corrected (zero)")
    plt.plot(x, y, linewidth=2, label="Mean per tyre age")
    plt.plot(x_smooth, y_fit, color="red", 
            label=f"Exp fit: y = {a:.3f} e^({b:.3f}x)")

    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Lap time (s)")
    plt.title(f"{COMPOUND} Lap Times vs Tyre Age (Fuel-corrected, Zero)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{COMPOUND}_scatter_mean_fuel_zero.png"), dpi=300)
    plt.close()


    print(f"Saved 4 plots for {COMPOUND}")
