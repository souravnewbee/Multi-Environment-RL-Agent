"""
migrate_qtables_to_db.py

One-time migration: reads your EXISTING trained .npy Q-tables from
qtables/ and imports them into qtables/qtables.db, so you don't have
to retrain everything from scratch just to switch storage backends.

Run this ONCE, after adding qtable_store.py and BEFORE deleting any
.npy files:

    python migrate_qtables_to_db.py

It's safe to run more than once — it just overwrites the DB entries
with the current .npy contents each time.
"""

import os
import glob
import numpy as np
from qtable_store import save_qtable

QTABLE_DIR = "qtables"

# Maps a filename pattern -> the DB name to store it under.
# Add/remove entries here if your filenames differ.
FILES_TO_MIGRATE = {
    "hospital_bed_allocation.npy":       "hospital_bed_allocation",
    "hospital_er_queue.npy":             "hospital_er_queue",
    "hospital_staff_allocation.npy":     "hospital_staff_allocation",

    "energy_solar_scheduling_qtable.npy":   "energy_solar_scheduling",
    "energy_battery_management_qtable.npy": "energy_battery_management",
    "energy_grid_interaction_qtable.npy":   "energy_grid_interaction",

    "traffic_intersection_qtable.npy": "traffic_intersection",
    "traffic_pedestrian_qtable.npy":   "traffic_pedestrian",
    "traffic_parking_qtable.npy":      "traffic_parking",

    "finance_trading.npy": "finance_trading",
    "finance_savings.npy": "finance_savings",
    "finance_budget.npy":  "finance_budget",

    "agriculture_soil_preparation.npy": "agriculture_soil_preparation",
    "agriculture_irrigation.npy":       "agriculture_irrigation",
    "agriculture_pest_control.npy":     "agriculture_pest_control",
}


def main():
    print("\n" + "=" * 60)
    print("  Migrating existing .npy Q-tables into qtables.db")
    print("=" * 60)

    migrated = 0
    skipped  = 0

    for filename, db_name in FILES_TO_MIGRATE.items():
        path = os.path.join(QTABLE_DIR, filename)
        if not os.path.exists(path):
            print(f"  [skip] {filename} — not found, nothing to migrate")
            skipped += 1
            continue

        Q = np.load(path)
        save_qtable(db_name, Q)
        print(f"  [✓] {filename}  →  qtables.db as '{db_name}'  (shape {Q.shape})")
        migrated += 1

    print(f"\n  Done. Migrated: {migrated}   Skipped (not found): {skipped}")

    # Catch any .npy files sitting in qtables/ that this script doesn't know about,
    # so nothing gets silently missed.
    all_npy   = set(os.path.basename(p) for p in glob.glob(os.path.join(QTABLE_DIR, "*.npy")))
    known_npy = set(FILES_TO_MIGRATE.keys()) | {f.replace(".npy", "_visits.npy") for f in FILES_TO_MIGRATE.keys()}
    unknown   = all_npy - known_npy
    if unknown:
        print(f"\n  [!] Found .npy files not in the migration list (left untouched):")
        for f in sorted(unknown):
            print(f"      {f}")
        print(f"  If any of these are real Q-tables, add them to FILES_TO_MIGRATE and re-run.")

    print(f"\n  Your trained tables are now also in qtables/qtables.db.")
    print(f"  Once you've confirmed the app reads correctly from the DB,")
    print(f"  the old .npy files can be safely deleted.\n")


if __name__ == "__main__":
    main()
