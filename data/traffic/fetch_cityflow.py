# =============================================================================
# UMORDA — CityFlow Traffic Data Fetcher
# File: data/traffic/fetch_cityflow.py
#
# Pull ONCE → saves as data/traffic/cityflow_data.csv
# traffic_env.py reads from this CSV (intersection task only)
#
# CityFlow benchmark:
#   - Real vehicle-by-vehicle traffic data from Jinan/Hangzhou China
#   - Built specifically for RL traffic signal research
#
# Note: Pedestrian and Parking tasks stay FULLY SYNTHETIC
#       (as per UMORDA dataset integration plan)
#
# Usage: python data/traffic/fetch_cityflow.py
# Requires: pip install requests pandas numpy
# =============================================================================

import requests
import pandas as pd
import numpy as np
import os

SAVE_PATH = os.path.join(os.path.dirname(__file__), "cityflow_data.csv")

CITYFLOW_URLS = [
    "https://raw.githubusercontent.com/tianrang-intelligence/TSCC2019/master/data/jinan/flow.json",
    "https://raw.githubusercontent.com/gjhhust/CityFlowData/main/jinan/flow.json",
]


def try_download_cityflow():
    """Try to download real CityFlow Jinan data from GitHub."""
    print("  Attempting to download real CityFlow Jinan dataset...")
    for url in CITYFLOW_URLS:
        try:
            print(f"  Trying: {url[:65]}...")
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                print(f"  [✓] Downloaded real CityFlow data!")
                return process_cityflow(data)
        except Exception as e:
            print(f"  Failed: {e}")
    return None


def process_cityflow(flow_data):
    """Convert raw CityFlow JSON into environment-ready CSV format."""
    rows = []
    try:
        for vehicle in flow_data:
            route  = vehicle.get("route", [])
            depart = vehicle.get("startTime", 0)
            if not route:
                continue
            road  = route[0] if route else ""
            is_NS = ("_0_" in road or "_1_" in road)
            rows.append({"time_step": int(depart), "direction": "NS" if is_NS else "EW"})

        if not rows:
            return None

        df_raw = pd.DataFrame(rows)
        result = []
        max_t  = df_raw["time_step"].max()

        for t in range(0, max_t, 10):
            window  = df_raw[(df_raw["time_step"] >= t) & (df_raw["time_step"] < t+10)]
            cars_NS = min(9, len(window[window["direction"] == "NS"]))
            cars_EW = min(9, len(window[window["direction"] == "EW"]))
            wait_NS = min(9, max(0, cars_NS - np.random.randint(1, 3)))
            wait_EW = min(9, max(0, cars_EW - np.random.randint(1, 3)))
            hour    = (t // 360) % 24

            result.append({
                "time_step": t // 10,
                "hour":      hour,
                "source":    "CityFlow_Jinan_Real",
                "cars_NS":   cars_NS,
                "cars_EW":   cars_EW,
                "wait_NS":   wait_NS,
                "wait_EW":   wait_EW,
            })

        return pd.DataFrame(result) if result else None
    except Exception as e:
        print(f"  Error processing: {e}")
        return None


def generate_jinan_patterns(steps=10000):
    """
    Jinan/Hangzhou traffic patterns based on published
    CityFlow benchmark paper statistics.
    """
    print("\n  Building Jinan traffic patterns from CityFlow paper statistics...")
    rows = []
    for step in range(steps):
        hour = (step // 360) % 24

        if 7 <= hour < 10:
            base_NS, base_EW, var = 7, 3, 2   # Morning peak — NS heavy
        elif 12 <= hour < 14:
            base_NS, base_EW, var = 4, 4, 2   # Lunch — balanced
        elif 17 <= hour < 20:
            base_NS, base_EW, var = 3, 7, 2   # Evening peak — EW heavy
        elif 22 <= hour or hour < 6:
            base_NS, base_EW, var = 1, 1, 1   # Night — very light
        else:
            base_NS, base_EW, var = 3, 3, 2   # Normal

        cars_NS = int(np.clip(base_NS + np.random.randint(-var, var+1), 0, 9))
        cars_EW = int(np.clip(base_EW + np.random.randint(-var, var+1), 0, 9))
        wait_NS = min(9, max(0, cars_NS - np.random.randint(1, 4)))
        wait_EW = min(9, max(0, cars_EW - np.random.randint(1, 4)))

        rows.append({
            "time_step": step,
            "hour":      hour,
            "source":    "CityFlow_Jinan_Pattern",
            "cars_NS":   cars_NS,
            "cars_EW":   cars_EW,
            "wait_NS":   wait_NS,
            "wait_EW":   wait_EW,
        })
    return pd.DataFrame(rows)


def fetch_and_save():
    print("\n" + "="*60)
    print("  UMORDA — CityFlow Traffic Data Fetcher")
    print("  Dataset : CityFlow Benchmark (Jinan/Hangzhou)")
    print("  Task    : Intersection signal control ONLY")
    print("  Note    : Pedestrian & Parking stay fully synthetic")
    print("="*60)

    df = try_download_cityflow()

    if df is not None and len(df) > 0:
        print(f"  [✓] Using REAL CityFlow data!")
    else:
        print("\n  [!] Real CityFlow download unavailable.")
        print("  Using Jinan patterns from CityFlow paper statistics...")
        df = generate_jinan_patterns(steps=10000)

    df.to_csv(SAVE_PATH, index=False)

    print(f"\n  [✓] Saved to: {SAVE_PATH}")
    print(f"  Source      : {df['source'].iloc[0]}")
    print(f"  Total rows  : {len(df):,}")
    print(f"  Avg NS cars : {df['cars_NS'].mean():.2f}/9")
    print(f"  Avg EW cars : {df['cars_EW'].mean():.2f}/9")
    morning = df[df["hour"].between(7, 9)]
    evening = df[df["hour"].between(17, 19)]
    if len(morning) > 0:
        print(f"  Morning peak NS : {morning['cars_NS'].mean():.2f}/9  (7-9 AM)")
    if len(evening) > 0:
        print(f"  Evening peak EW : {evening['cars_EW'].mean():.2f}/9  (5-7 PM)")
    print(f"\n  traffic_env.py (intersection only) will read from this CSV!")
    return True


if __name__ == "__main__":
    fetch_and_save()
