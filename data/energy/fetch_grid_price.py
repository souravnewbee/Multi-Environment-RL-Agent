# =============================================================================
# UMORDA — BPDB Grid Price CSV Creator
# File: data/energy/fetch_grid_price.py
#
# Creates BPDB tariff schedule → saves as data/energy/grid_price.csv
# energy_env.py reads from this CSV every training run
#
# Source: Bangladesh Power Development Board published tariff slabs
#         (peak/off-peak/normal rates)
#
# Usage: python data/energy/fetch_grid_price.py
# =============================================================================

import pandas as pd
import os

SAVE_PATH = os.path.join(os.path.dirname(__file__), "grid_price.csv")


def create_and_save():
    print("\n" + "="*60)
    print("  UMORDA — BPDB Grid Price Schedule")
    print("  Source: Bangladesh Power Development Board")
    print("  Tariff: Peak / Normal / Off-peak rates")
    print("="*60)

    # Real BPDB tariff schedule
    # Peak:     17:00 - 23:00  → most expensive
    # Normal:   06:00 - 17:00  → standard rate
    # Off-peak: 23:00 - 06:00  → cheapest
    rows = []
    for hour in range(24):
        if 23 <= hour or hour < 6:
            rows.append({
                "hour":        hour,
                "price_level": 0,               # 0 = Cheap (for Q-table)
                "price_label": "Off-peak",
                "price_bdt":   5.13,            # BDT per kWh
                "period":      "23:00 - 06:00",
            })
        elif 17 <= hour < 23:
            rows.append({
                "hour":        hour,
                "price_level": 2,               # 2 = Expensive (for Q-table)
                "price_label": "Peak",
                "price_bdt":   11.36,
                "period":      "17:00 - 23:00",
            })
        else:
            rows.append({
                "hour":        hour,
                "price_level": 1,               # 1 = Normal (for Q-table)
                "price_label": "Normal",
                "price_bdt":   7.98,
                "period":      "06:00 - 17:00",
            })

    df = pd.DataFrame(rows)
    df.to_csv(SAVE_PATH, index=False)

    print(f"\n  [✓] Saved to: {SAVE_PATH}")
    print(f"\n  {'Hour':>5} | {'Level':>6} | {'Label':>10} | {'Rate (BDT/kWh)':>14} | Period")
    print(f"  {'─'*60}")
    for _, row in df.iterrows():
        bar = ["█", "██", "███"][int(row["price_level"])]
        print(f"  {int(row['hour']):>5} | {bar:<6} | {row['price_label']:>10} | "
              f"{row['price_bdt']:>14.2f} | {row['period']}")

    print(f"\n  energy_env.py will now read grid_price from this CSV!")
    return True


if __name__ == "__main__":
    create_and_save()
