# collect_and_plot_all_tyres.py

import os
import json
import time
import requests
import certifi
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_and_cache(url, fname, sleep=0.05):
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    r = requests.get(url, verify=certifi.where(), timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(path, "w") as f:
        json.dump(data, f)
    time.sleep(sleep)
    return data

# USER PARAMETERS
COUNTRY = "Austria"
YEAR = 2025
SESSION_TYPE = "Practice"
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]   # process all
ANOMALY_THRESHOLD = 74.0
FUEL_PER_LAP_KG = 1.5
SECONDS_SAVED_PER_1p5kg = 0.1

# --- get sessions ---
sessions_url = f"https://api.openf1.org/v1/sessions?country_name={COUNTRY}&session_type={SESSION_TYPE}&year={YEAR}"
sessions = fetch_and_cache(sessions_url, f"sessions_{COUNTRY}_{SESSION_TYPE}_{YEAR}.json")
practice_session_keys = [s["session_key"] for s in sessions[:3]]
print("Sessions to process:", practice_session_keys)

for COMPOUND in COMPOUNDS:
    print(f"\n===== Processing compound: {COMPOUND} =====")
    rows = []

    for sesh in practice_session_keys:
        print(f"  Session {sesh} ...")

        stints = fetch_and_cache(f"https://api.openf1.org/v1/stints?session_key={sesh}",
                                 f"stints_{sesh}.json")
        laps = fetch_and_cache(f"https://api.openf1.org/v1/laps?session_key={sesh}",
                               f"laps_{sesh}.json")

        # index laps
        laps_by_driver = defaultdict(dict)
        for lap in laps:
            dnum = lap.get("driver_number")
            ln = lap.get("lap_number")
            if dnum is not None and ln is not None:
                laps_by_driver[dnum][ln] = lap

        # iterate stints
        for st in stints:
            comp = st.get("compound")
            if not comp or comp.upper() != COMPOUND.upper():
                continue
            dnum = st.get("driver_number")
            try:
                start = int(st.get("lap_start"))
                end = int(st.get("lap_end"))
            except (TypeError, ValueError):
                continue
            stint_length = max(0, end - start)
            tyre_age_start = int(st.get("tyre_age_at_start", 0))

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
                    "session": sesh,
                    "lap_number": ln,
                    "stint_start": start,
                    "stint_end": end,
                    "stint_length": stint_length,
                    "tyre_age_at_start": tyre_age_start
                })

    if not rows:
        print(f"⚠️ No data found for {COMPOUND}, skipping plots.")
        continue

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

        penalty_sec = remaining_fuel_laps * SECONDS_SAVED_PER_1p5kg
        fuel_corrected_time_zero = r["lap_time"] - penalty_sec

        baseline_remaining = max(0, baseline_start_fuel - laps_done)
        penalty_baseline = baseline_remaining * SECONDS_SAVED_PER_1p5kg
        fuel_corrected_time_std = r["lap_time"] - (penalty_sec - penalty_baseline)

        newrow = r.copy()
        newrow.update({
            "start_fuel_laps": start_fuel_laps,
            "remaining_fuel_laps": remaining_fuel_laps,
            "penalty_seconds": penalty_sec,
            "fuel_corrected_time_zero": fuel_corrected_time_zero,
            "fuel_corrected_time_std": fuel_corrected_time_std
        })
        detailed.append(newrow)

    # --- filter anomalies ---
    lap_times = np.array([r["lap_time"] for r in detailed])
    ages = np.array([r["tyre_age"] for r in detailed])
    fuel_zero = np.array([r["fuel_corrected_time_zero"] for r in detailed])
    fuel_std = np.array([r["fuel_corrected_time_std"] for r in detailed])

    mask = (lap_times <= ANOMALY_THRESHOLD) & np.isfinite(lap_times)
    lap_times_clean = lap_times[mask]
    ages_clean = ages[mask]
    fuel_zero_clean = fuel_zero[mask]
    fuel_std_clean = fuel_std[mask]

    print(f"  After filtering lap_time <= {ANOMALY_THRESHOLD}: {len(lap_times_clean)} rows remain.")

    # --- plotting ---
    unique_ages = sorted(set(ages_clean))
    mean_times_raw = [lap_times_clean[ages_clean == a].mean() for a in unique_ages]
    mean_times_fuel_zero = [fuel_zero_clean[ages_clean == a].mean() for a in unique_ages]
    mean_times_fuel_std = [fuel_std_clean[ages_clean == a].mean() for a in unique_ages]

    # A: raw scatter
    plt.figure(figsize=(10,6))
    plt.scatter(ages_clean, lap_times_clean, alpha=0.5, s=10)
    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Lap time (s)")
    plt.title(f"{COMPOUND} Lap Times vs Tyre Age (Cleaned)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, f"{COMPOUND}_scatter_raw.png"), dpi=300)
    plt.close()

    # B: raw + mean
    plt.figure(figsize=(10,6))
    plt.scatter(ages_clean, lap_times_clean, alpha=0.4, s=10, label="Lap times")
    plt.plot(unique_ages, mean_times_raw, linewidth=2, label="Mean per tyre age")
    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Lap time (s)")
    plt.title(f"{COMPOUND} Lap Times vs Tyre Age (Mean)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, f"{COMPOUND}_scatter_mean_raw.png"), dpi=300)
    plt.close()

    # C: fuel corrected (zero baseline)
    plt.figure(figsize=(10,6))
    plt.scatter(ages_clean, fuel_zero_clean, alpha=0.4, s=10, label="Fuel-corrected (zero)")
    plt.plot(unique_ages, mean_times_fuel_zero, linewidth=2, label="Mean per tyre age")
    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Lap time (s)")
    plt.title(f"{COMPOUND} Lap Times vs Tyre Age (Fuel-corrected, Zero)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, f"{COMPOUND}_scatter_mean_fuel_zero.png"), dpi=300)
    plt.close()

    # D: fuel corrected (baseline)
    plt.figure(figsize=(10,6))
    plt.scatter(ages_clean, fuel_std_clean, alpha=0.4, s=10,
                label=f"Fuel-corrected (baseline={baseline_start_fuel} laps)")
    plt.plot(unique_ages, mean_times_fuel_std, linewidth=2, label="Mean per tyre age")
    plt.xlabel("Tyre age (laps)")
    plt.ylabel("Lap time (s)")
    plt.title(f"{COMPOUND} Lap Times vs Tyre Age (Fuel-corrected, Baseline)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, f"{COMPOUND}_scatter_mean_fuel_baseline.png"), dpi=300)
    plt.close()

    print(f"  ✅ Saved 4 plots for {COMPOUND}")

# ============================
# Overlay plot across compounds
# ============================
print("\n===== Generating overlay plot across compounds =====")

def moving_average(x, y, window=5):
    order = np.argsort(x)
    x_sorted, y_sorted = np.array(x)[order], np.array(y)[order]
    if len(y_sorted) < window:
        return x_sorted, y_sorted  # not enough points to smooth
    y_smooth = np.convolve(y_sorted, np.ones(window)/window, mode="valid")
    x_smooth = x_sorted[:len(y_smooth)]
    return x_smooth, y_smooth

tyre_colors = {"SOFT": "red", "MEDIUM": "yellow", "HARD": "grey"}
plt.figure(figsize=(10,6))

for COMPOUND in COMPOUNDS:
    # load back processed laps from cache results
    # we already have rows built earlier, but only per compound loop
    # easiest way: reload from cache outputs
    all_rows = []
    for sesh in practice_session_keys:
        stints = fetch_and_cache(f"https://api.openf1.org/v1/stints?session_key={sesh}",
                                 f"stints_{sesh}.json")
        laps = fetch_and_cache(f"https://api.openf1.org/v1/laps?session_key={sesh}",
                               f"laps_{sesh}.json")

        laps_by_driver = defaultdict(dict)
        for lap in laps:
            dnum = lap.get("driver_number")
            ln = lap.get("lap_number")
            if dnum is not None and ln is not None:
                laps_by_driver[dnum][ln] = lap

        for st in stints:
            comp = st.get("compound")
            if not comp or comp.upper() != COMPOUND.upper():
                continue
            dnum = st.get("driver_number")
            try:
                start = int(st.get("lap_start"))
                end = int(st.get("lap_end"))
            except (TypeError, ValueError):
                continue
            tyre_age_start = int(st.get("tyre_age_at_start", 0))
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
                all_rows.append((ln, lap_time, tyre_age))

    if not all_rows:
        continue

    lap_numbers = np.array([r[0] for r in all_rows])
    lap_times = np.array([r[1] for r in all_rows])

    # filter anomalies
    mask = (lap_times <= ANOMALY_THRESHOLD) & np.isfinite(lap_times)
    lap_numbers = lap_numbers[mask]
    lap_times = lap_times[mask]

    if len(lap_numbers) == 0:
        continue

    x_smooth, y_smooth = moving_average(lap_numbers, lap_times, window=5)
    plt.plot(x_smooth, y_smooth, color=tyre_colors.get(COMPOUND, "black"),
             linewidth=2, label=COMPOUND)

plt.title("Overlayed Tyre Degradation Curves (All Compounds)")
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (s)")
plt.legend(title="Tyre Compound")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(CACHE_DIR, "overlay_tyres.png"), dpi=300)
plt.close()

print("  ✅ Saved overlay plot: overlay_tyres.png")
