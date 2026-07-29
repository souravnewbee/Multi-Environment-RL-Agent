# =============================================================================
# UMORDA — NASA POWER Solar Data Fetcher
# File: data/energy/fetch_solar.py
#
# Pull ONCE → saves as data/energy/solar_data.csv
# energy_env.py reads from this CSV every training run
#
# NASA POWER API:
#   - Completely FREE, no API key needed
#   - Real hourly solar irradiance for Dhaka, Bangladesh
#   - URL: https://power.larc.nasa.gov
#
# Usage: python data/energy/fetch_solar.py
# Requires: pip install requests pandas
# =============================================================================

import requests
import pandas as pd
import os
from datetime import datetime

# ── Save location ──────────────────────────────────────────────────────────────
SAVE_PATH = os.path.join(os.path.dirname(__file__), "solar_data.csv")

# ── Dhaka, Bangladesh ──────────────────────────────────────────────────────────
LATITUDE  = 23.8103
LONGITUDE = 90.4125

# ── NASA POWER API ─────────────────────────────────────────────────────────────
NASA_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_PARAMS = {
    "parameters":    "ALLSKY_SFC_SW_DWN,T2M",  # solar irradiance W/m² + temperature °C
    "community":     "RE",                       # Renewable Energy community
    "longitude":     LONGITUDE,
    "latitude":      LATITUDE,
    "start":         "20230101",                 # full year 2023
    "end":           "20231231",
    "format":        "JSON",
    "time-standard": "LST",                      # Local Standard Time Dhaka
}


def fetch_and_save():
    print("\n" + "="*60)
    print("  UMORDA — NASA POWER Solar Data Fetcher")
    print("  Location : Dhaka, Bangladesh")
    print("  Period   : 2023 (hourly — 8,760 data points)")
    print("  API      : https://power.larc.nasa.gov (FREE)")
    print("="*60)

    print("\n  Sending request to NASA POWER...")
    print("  (This may take 20-60 seconds...)\n")

    try:
        response = requests.get(NASA_URL, params=NASA_PARAMS, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Extract hourly parameters
        props      = data["properties"]["parameter"]
        solar_dict = props["ALLSKY_SFC_SW_DWN"]   # W/m²
        temp_dict  = props["T2M"]                   # °C

        print(f"  [✓] NASA POWER responded!")
        print(f"  Processing {len(solar_dict)} hourly records...\n")

        rows = []
        for ts_str, irr in solar_dict.items():
            # Parse timestamp (format: YYYYMMDDHMM or YYYYMMDDHH)
            try:
                dt = datetime.strptime(ts_str, "%Y%m%d%H%M")
            except:
                try:
                    dt = datetime.strptime(ts_str[:10], "%Y%m%d%H")
                except:
                    continue

            hour = dt.hour
            irr  = max(0.0, irr if irr is not None else 0.0)
            temp = temp_dict.get(ts_str, 28.0)

            # Scale irradiance to 0-9 for Q-table
            # Bangladesh peak ~900 W/m²
            solar_scaled = min(9, int(irr / 900 * 9))

            # Time of day
            if 6 <= hour < 12:    time_of_day = 0   # Morning
            elif 12 <= hour < 17: time_of_day = 1   # Afternoon (peak sun)
            elif 17 <= hour < 21: time_of_day = 2   # Evening
            else:                 time_of_day = 3   # Night

            rows.append({
                "timestamp":           dt.strftime("%Y-%m-%d %H:%M"),
                "hour":                hour,
                "solar_irradiance_wm2": irr,
                "solar_output_scaled": solar_scaled,   # 0-9 for Q-table
                "temperature_c":       temp,
                "time_of_day":         time_of_day,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(SAVE_PATH, index=False)

        # Summary
        day_df = df[df["solar_irradiance_wm2"] > 0]
        print(f"  [✓] Saved to: {SAVE_PATH}")
        print(f"  Total rows         : {len(df):,}")
        print(f"  Daytime rows       : {len(day_df):,}")
        print(f"  Avg irradiance     : {day_df['solar_irradiance_wm2'].mean():.1f} W/m²")
        print(f"  Max irradiance     : {df['solar_irradiance_wm2'].max():.1f} W/m²")
        print(f"  Avg scaled output  : {df['solar_output_scaled'].mean():.2f}/9")
        print(f"\n  energy_env.py will now read from this CSV!")
        return True

    except requests.exceptions.ConnectionError:
        print("  [✗] Cannot connect to NASA POWER API!")
        print("  Check your internet connection and try again.")
        return False
    except requests.exceptions.Timeout:
        print("  [✗] Request timed out! NASA server may be busy.")
        print("  Try again in a few minutes.")
        return False
    except Exception as e:
        print(f"  [✗] Error: {e}")
        return False


if __name__ == "__main__":
    success = fetch_and_save()
    if not success:
        print("\n  [!] Solar data fetch failed.")
        print("  Fix the error above and run again.")
        print("  energy_env.py will use random fallback until CSV is ready.")
