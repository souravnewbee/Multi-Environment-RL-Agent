# =============================================================================
# UMORDA — Traffic Domain: Human-Readable Q-Table Viewer (CLEAN VERSION)
# File: view_qtable_traffic.py
# Usage: python view_qtable_traffic.py
# =============================================================================

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))
from environments.traffic_env import TrafficEnv

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "qtables")

# =============================================================================
# STATE DECODER
# =============================================================================
def decode_state(state_idx: int, bins: list) -> list:
    values = []
    for b in reversed(bins):
        values.append(state_idx % b)
        state_idx //= b
    return list(reversed(values))


# =============================================================================
# INTERSECTION — clean block format
# =============================================================================
def print_intersection(qtable, show_limit=999):
    cfg        = TrafficEnv.TASK_CONFIG["intersection"]
    bins       = cfg["state_bins"]
    n_states   = int(np.prod(bins))
    actions    = cfg["action_meanings"]

    print("\n")
    print("=" * 72)
    print("  INTERSECTION Q-TABLE — Single Traffic Intersection Control")
    print("  Actions:  [0] Green NS (open North-South)")
    print("            [1] Green EW (open East-West)")
    print("  Reward  : Agent earns more reward by clearing the busier direction")
    print("=" * 72)

    count = 0
    for s in range(n_states):
        q = qtable[s]
        if np.all(q == 0):
            continue

        n, s2, e, w = decode_state(s, bins)
        ns = n + s2
        ew = e + w
        best = int(np.argmax(q))

        # Traffic level
        total = ns + ew
        if total == 0:
            level = "Empty road"
        elif total <= 3:
            level = "Light traffic"
        elif total <= 6:
            level = "Moderate traffic"
        else:
            level = "Heavy traffic"

        # Dominant direction
        if ns > ew:
            dominant = f"North-South busier  (NS={ns} vs EW={ew})"
        elif ew > ns:
            dominant = f"East-West busier    (NS={ns} vs EW={ew})"
        else:
            dominant = f"Both equal          (NS={ns} = EW={ew})"

        # Confidence
        diff = abs(q[0] - q[1])
        if diff > 10:
            confidence = "HIGH confidence"
        elif diff > 3:
            confidence = "MODERATE confidence"
        else:
            confidence = "LOW confidence"

        print(f"  ┌─ State: N={n} S={s2} E={e} W={w}")
        print(f"  │  Situation  : {level} — {dominant}")
        print(f"  │  Q[Green NS]: {q[0]:>8.3f}")
        print(f"  │  Q[Green EW]: {q[1]:>8.3f}")
        print(f"  │  DECISION   : ★ {actions[best]}  ({confidence})")
        print(f"  └─────────────────────────────────────────────────")

        count += 1
        if count >= show_limit:
            remaining = sum(1 for i in range(s+1, n_states)
                            if not np.all(qtable[i] == 0))
            if remaining:
                print(f"\n  ... {remaining} more states not shown (use show_limit)")
            break

    print(f"\n  Total trained states shown: {count}")
    print("=" * 72)


# =============================================================================
# PEDESTRIAN — clean block format
# =============================================================================
def print_pedestrian(qtable, show_limit=999):
    cfg      = TrafficEnv.TASK_CONFIG["pedestrian"]
    bins     = cfg["state_bins"]
    n_states = int(np.prod(bins))
    actions  = cfg["action_meanings"]

    print("\n")
    print("=" * 72)
    print("  PEDESTRIAN Q-TABLE — Pedestrian Crossing Control")
    print("  Actions:  [0] Allow Pedestrians to cross")
    print("            [1] Allow Vehicles to pass")
    print("  Reward  : Safety first — high pedestrian wait = penalty if ignored")
    print("=" * 72)

    count = 0
    for s in range(n_states):
        q = qtable[s]
        if np.all(q == 0):
            continue

        peds, vehs = decode_state(s, bins)
        best = int(np.argmax(q))

        if peds == 0 and vehs == 0:
            situation = "Empty crossing — nobody waiting"
        elif peds == 0:
            situation = f"No pedestrians — {vehs} vehicle(s) waiting"
        elif vehs == 0:
            situation = f"{peds} pedestrian(s) waiting — no vehicles"
        elif peds > vehs:
            situation = f"More pedestrians ({peds}) than vehicles ({vehs}) — safety priority"
        elif vehs > peds:
            situation = f"More vehicles ({vehs}) than pedestrians ({peds})"
        else:
            situation = f"Equal wait — {peds} peds and {vehs} vehs"

        safety = "⚠ HIGH RISK" if peds >= 4 else "OK"
        diff   = abs(q[0] - q[1])
        confidence = "HIGH" if diff > 5 else "MODERATE" if diff > 1 else "LOW"

        print(f"  ┌─ State: Pedestrians={peds}  Vehicles={vehs}  [{safety}]")
        print(f"  │  Situation        : {situation}")
        print(f"  │  Q[Allow Peds]    : {q[0]:>8.3f}")
        print(f"  │  Q[Allow Vehicles]: {q[1]:>8.3f}")
        print(f"  │  DECISION         : ★ {actions[best]}  ({confidence} confidence)")
        print(f"  └─────────────────────────────────────────────────")

        count += 1
        if count >= show_limit:
            break

    print(f"\n  Total trained states shown: {count}")
    print("=" * 72)


# =============================================================================
# PARKING — clean block format
# =============================================================================
def print_parking(qtable, show_limit=999):
    cfg      = TrafficEnv.TASK_CONFIG["parking"]
    bins     = cfg["state_bins"]
    n_states = int(np.prod(bins))
    actions  = cfg["action_meanings"]

    print("\n")
    print("=" * 72)
    print("  PARKING Q-TABLE — Parking Lot Management")
    print("  Actions:  [0] Open Zone A")
    print("            [1] Open Zone B")
    print("            [2] Close Entry")
    print("  Reward  : Maximize occupancy, avoid congestion overflow")
    print("=" * 72)

    count = 0
    for s in range(n_states):
        q = qtable[s]
        if np.all(q == 0):
            continue

        spots, incoming = decode_state(s, bins)
        best = int(np.argmax(q))

        if spots == 0:
            situation = f"LOT FULL — {incoming} vehicle(s) still incoming!"
        elif incoming == 0:
            situation = f"No vehicles incoming — {spots} spot(s) free"
        elif incoming > spots:
            situation = f"OVERFLOW RISK — {incoming} incoming > {spots} spots left"
        elif spots >= 8:
            situation = f"Plenty of space — {spots} spots free, {incoming} incoming"
        else:
            situation = f"Normal load — {spots} spots free, {incoming} incoming"

        print(f"  ┌─ State: Available Spots={spots}  Incoming Vehicles={incoming}")
        print(f"  │  Situation    : {situation}")
        print(f"  │  Q[Open ZoneA]: {q[0]:>8.3f}")
        print(f"  │  Q[Open ZoneB]: {q[1]:>8.3f}")
        print(f"  │  Q[Close Entr]: {q[2]:>8.3f}")
        print(f"  │  DECISION     : ★ {actions[best]}")
        print(f"  └─────────────────────────────────────────────────")

        count += 1
        if count >= show_limit:
            break

    print(f"\n  Total trained states shown: {count}")
    print("=" * 72)


# =============================================================================
# LOAD Q-TABLE HELPER
# =============================================================================
def load_qtable(task: str):
    path = os.path.join(QTABLE_DIR, f"traffic_{task}_qtable.npy")
    if not os.path.exists(path):
        print(f"\n  [!] Q-Table not found for '{task}'.")
        print(f"      Please run:  python training/train_traffic.py  first!\n")
        return None
    return np.load(path)


# =============================================================================
# SAVE REPORT TO TXT
# =============================================================================
def save_report(task: str):
    import io, contextlib
    qtable = load_qtable(task)
    if qtable is None:
        return

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if task == "intersection":
            print_intersection(qtable)
        elif task == "pedestrian":
            print_pedestrian(qtable)
        elif task == "parking":
            print_parking(qtable)

    path = os.path.join(QTABLE_DIR, f"traffic_{task}_qtable_explained.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [✓] Saved → {path}")


# =============================================================================
# MAIN MENU
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  UMORDA — TRAFFIC Q-TABLE VIEWER (Clean Format)")
    print("=" * 60)

    tasks_fn = {
        "intersection": print_intersection,
        "pedestrian":   print_pedestrian,
        "parking":      print_parking,
    }

    while True:
        print("\n  [1] Intersection Q-Table")
        print("  [2] Pedestrian Q-Table")
        print("  [3] Parking Q-Table")
        print("  [4] View ALL")
        print("  [5] Save ALL as .txt reports")
        print("  [q] Quit")

        choice = input("\n  Your choice: ").strip().lower()

        if choice == "1":
            q = load_qtable("intersection")
            if q is not None: print_intersection(q)
        elif choice == "2":
            q = load_qtable("pedestrian")
            if q is not None: print_pedestrian(q)
        elif choice == "3":
            q = load_qtable("parking")
            if q is not None: print_parking(q)
        elif choice == "4":
            for task, fn in tasks_fn.items():
                q = load_qtable(task)
                if q is not None: fn(q)
        elif choice == "5":
            for task in tasks_fn:
                save_report(task)
            print("\n  All reports saved in qtables/ folder!")
        elif choice == "q":
            print("\n  Goodbye! 👋\n")
            break
        else:
            print("  [!] Invalid choice.")

if __name__ == "__main__":
    main()
