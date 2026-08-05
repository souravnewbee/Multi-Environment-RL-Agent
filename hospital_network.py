"""
hospital_network.py

Prototype "hospital suggestion network" — when THIS hospital's agent
recommends Transfer/Reject because it has no capacity, this module finds
the nearest CONNECTED hospital that currently has room.

For the demo/showcase, hospitals are simulated (name, location, live bed
count that drifts randomly over time) rather than pulling from real
hospital admin systems — see report_status() docstring for what a real
integration would replace this with.

Usage:
    from hospital_network import suggest_nearby_hospital, seed_demo_network

    seed_demo_network()   # call once at startup to populate mock hospitals
    suggestion = suggest_nearby_hospital(free_beds_here=0)
"""

import random
from network_store import report_status, find_nearest_with_capacity

# This hospital's own identity in the network (used to exclude itself
# from its own "nearby hospital" suggestions, and to report its own
# live status so OTHER hospitals could find it too).
THIS_HOSPITAL_ID   = "H1"
THIS_HOSPITAL_NAME = "This Hospital (UMORDA)"
THIS_HOSPITAL_LAT  = 23.8103   # Dhaka -- change to your real hospital's location
THIS_HOSPITAL_LON  = 90.4125

# ── Demo/mock hospitals in the "network" ─────────────────────────────────
# Real deployment note: each of these would instead be a separate hospital
# running its own UMORDA instance, periodically calling report_status()
# with its own live free_beds/free_er_beds count (e.g. pulled from its
# admin system) — not randomly generated here.
DEMO_HOSPITALS = [
    {"id": "H2", "name": "City General Hospital",  "lat": 23.8210, "lon": 90.4050},
    {"id": "H3", "name": "Green Life Hospital",     "lat": 23.7500, "lon": 90.3900},
    {"id": "H4", "name": "Square Hospital",         "lat": 23.7550, "lon": 90.3700},
    {"id": "H5", "name": "United Hospital",         "lat": 23.7930, "lon": 90.4150},
]


def seed_demo_network(this_hospital_free_beds: int = 0, this_hospital_free_er: int = 0):
    """
    Populates network_store's DB with THIS hospital plus a handful of
    demo hospitals with randomised (but plausible) live bed counts.
    Call this once at the start of a session/demo.
    """
    report_status(
        THIS_HOSPITAL_ID, THIS_HOSPITAL_NAME,
        THIS_HOSPITAL_LAT, THIS_HOSPITAL_LON,
        free_beds=this_hospital_free_beds, free_er_beds=this_hospital_free_er,
    )
    for h in DEMO_HOSPITALS:
        report_status(
            h["id"], h["name"], h["lat"], h["lon"],
            free_beds=random.randint(0, 6),
            free_er_beds=random.randint(0, 3),
        )


def update_this_hospital_status(free_beds: int, free_er_beds: int = 0):
    """Call whenever this hospital's own bed counts change, to keep the
    network's view of THIS hospital current."""
    report_status(
        THIS_HOSPITAL_ID, THIS_HOSPITAL_NAME,
        THIS_HOSPITAL_LAT, THIS_HOSPITAL_LON,
        free_beds=free_beds, free_er_beds=free_er_beds,
    )


def suggest_nearby_hospital(need_er: bool = True) -> dict:
    """
    Returns the nearest connected hospital with capacity, or None if
    nobody in the network currently has room.

    Returns a dict like:
        {"hospital_id": "H2", "name": "City General Hospital",
         "free_beds": 3, "free_er_beds": 1, "distance_km": 1.4, ...}
    or None.
    """
    return find_nearest_with_capacity(
        THIS_HOSPITAL_LAT, THIS_HOSPITAL_LON,
        need_er=need_er, exclude_id=THIS_HOSPITAL_ID,
    )


def format_suggestion(suggestion: dict, need_er: bool = True, reason: str = "Out of ER beds") -> str:
    """Short, direct line — not meant to be fed into an LLM explanation."""
    if suggestion is None:
        return f"{reason}. No connected hospital currently has capacity."
    count = suggestion["free_er_beds"] if need_er else suggestion["free_beds"]
    return (f"{reason}. Nearest hospital: {suggestion['name']} — "
            f"{count} ER beds available ({suggestion['distance_km']} km away).")


if __name__ == "__main__":
    seed_demo_network(this_hospital_free_beds=0, this_hospital_free_er=0)
    print("Network seeded.\n")

    s = suggest_nearby_hospital()
    print("Suggestion when out of ER beds:")
    print(" ", format_suggestion(s))