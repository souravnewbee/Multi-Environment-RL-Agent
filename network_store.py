"""
network_store.py

Storage layer for LIVE hospital bed availability — separate on purpose
from qtable_store.py, which stores TRAINED Q-tables.

Why separate:
  - qtables.db  : written once by a trainer, read many times. Never changes
                  once trained.
  - network.db  : written frequently (every hospital reports its live
                  status), read by anyone looking for capacity. Genuinely
                  different access pattern, so it gets its own file.

This is intentionally simple (raw sqlite3, no ORM) since the schema is
small and single-table. If this grows into a real multi-hospital,
multi-writer service later, this is the piece that would move to a
hosted database (Postgres) — not qtable_store.py.
"""

import sqlite3
import os
import math
import datetime

DEFAULT_DB_PATH = os.path.join("qtables", "network.db")   # kept near qtables/ but a separate file


def _connect(db_path=DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hospital_status (
            hospital_id   TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            lat           REAL NOT NULL,
            lon           REAL NOT NULL,
            free_beds     INTEGER NOT NULL,
            free_er_beds  INTEGER NOT NULL,
            updated_at    TEXT NOT NULL
        )
    """)
    return conn


def report_status(hospital_id: str, name: str, lat: float, lon: float,
                   free_beds: int, free_er_beds: int, db_path: str = DEFAULT_DB_PATH):
    """
    Called by a hospital to push its current live status.
    Overwrites any previous status for that hospital_id.
    """
    conn = _connect(db_path)
    conn.execute(
        """INSERT INTO hospital_status
               (hospital_id, name, lat, lon, free_beds, free_er_beds, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(hospital_id) DO UPDATE SET
               name=excluded.name, lat=excluded.lat, lon=excluded.lon,
               free_beds=excluded.free_beds, free_er_beds=excluded.free_er_beds,
               updated_at=excluded.updated_at""",
        (hospital_id, name, lat, lon, free_beds, free_er_beds,
         datetime.datetime.now(datetime.timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def get_all_hospitals(db_path: str = DEFAULT_DB_PATH) -> list:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT hospital_id, name, lat, lon, free_beds, free_er_beds, updated_at FROM hospital_status"
    ).fetchall()
    conn.close()
    return [
        {"hospital_id": r[0], "name": r[1], "lat": r[2], "lon": r[3],
         "free_beds": r[4], "free_er_beds": r[5], "updated_at": r[6]}
        for r in rows
    ]


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance between two lat/lon points, in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_nearest_with_capacity(lat: float, lon: float, need_er: bool = False,
                                exclude_id: str = None, db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Returns the nearest hospital (by straight-line distance) that currently
    has capacity, or None if nobody in the network has room.

    need_er=True filters on free_er_beds instead of free_beds.
    """
    candidates = get_all_hospitals(db_path)
    best, best_dist = None, None

    for h in candidates:
        if exclude_id and h["hospital_id"] == exclude_id:
            continue
        capacity = h["free_er_beds"] if need_er else h["free_beds"]
        if capacity <= 0:
            continue
        dist = _haversine_km(lat, lon, h["lat"], h["lon"])
        if best is None or dist < best_dist:
            best, best_dist = h, dist

    if best is None:
        return None
    return {**best, "distance_km": round(best_dist, 1)}


if __name__ == "__main__":
    # Quick self-test
    report_status("H1", "Self (this hospital)", 23.8103, 90.4125, free_beds=0, free_er_beds=0)
    report_status("H2", "City General",          23.8210, 90.4050, free_beds=3, free_er_beds=1)
    report_status("H3", "Green Life Hospital",    23.7500, 90.3900, free_beds=0, free_er_beds=2)

    print("All hospitals:")
    for h in get_all_hospitals():
        print(" ", h)

    print("\nNearest with a free bed (excluding H1):")
    print(" ", find_nearest_with_capacity(23.8103, 90.4125, exclude_id="H1"))
