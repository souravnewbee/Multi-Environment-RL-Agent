# =============================================================================
# UMORDA — Traffic Domain: Human-Readable Q-Table Viewer (FIXED v2)
# File: view_qtable_traffic.py
# =============================================================================

import sys, os, numpy as np, io, contextlib
sys.path.append(os.path.dirname(__file__))
from environments.traffic_env import TrafficEnv

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "qtables")

def load_qtable(task):
    path = os.path.join(QTABLE_DIR, f"traffic_{task}_qtable.npy")
    if not os.path.exists(path):
        print(f"  [!] Q-Table not found. Run: python training/train_traffic.py first!")
        return None
    return np.load(path)

def decode_state(state_idx, bins):
    values = []
    for b in reversed(bins):
        values.append(state_idx % b)
        state_idx //= b
    return list(reversed(values))

# =============================================================================
# INTERSECTION
# =============================================================================
def print_intersection(qtable, limit=50):
    cfg      = TrafficEnv.TASK_CONFIG["intersection"]
    bins     = cfg["state_bins"]
    n_states = int(np.prod(bins))
    actions  = cfg["action_meanings"]

    print("\n" + "="*75)
    print("  INTERSECTION Q-TABLE — Single Traffic Intersection Control (FIXED v2)")
    print("  State variables: cars_NS, cars_EW, current_phase,")
    print("                   phase_elapsed, max_wait_NS, max_wait_EW")
    print("  Actions: [0] Green NS   [1] Green EW")
    print("  Safety : Hard override if any direction waits > 8 steps")
    print("="*75)

    count = 0
    for s in range(n_states):
        q = qtable[s]
        if np.all(np.abs(q - 1.0) < 0.01): continue  # skip untrained

        cars_NS, cars_EW, phase, elapsed, wait_NS, wait_EW = decode_state(s, bins)
        best = int(np.argmax(q))

        total = cars_NS + cars_EW
        if total == 0:   level = "Empty road"
        elif total <= 4: level = "Light traffic"
        elif total <= 8: level = "Moderate traffic"
        else:            level = "Heavy traffic"

        if wait_NS > wait_EW: urgency = f"NS more urgent (waited {wait_NS} steps)"
        elif wait_EW > wait_NS: urgency = f"EW more urgent (waited {wait_EW} steps)"
        else:                   urgency = f"Equal urgency (both waited {wait_NS} steps)"

        phase_str = "Green NS active" if phase == 0 else "Green EW active"
        safety_ns = " ⚠ NEAR LIMIT!" if wait_NS >= 6 else ""
        safety_ew = " ⚠ NEAR LIMIT!" if wait_EW >= 6 else ""

        diff = abs(q[0] - q[1])
        conf = "HIGH" if diff > 5 else "MODERATE" if diff > 2 else "LOW"

        print(f"  ┌─ NS={cars_NS} cars, EW={cars_EW} cars | {level}")
        print(f"  │  Wait times   : NS waited {wait_NS} steps{safety_ns} | EW waited {wait_EW} steps{safety_ew}")
        print(f"  │  Urgency      : {urgency}")
        print(f"  │  Current phase: {phase_str} (active for {elapsed} steps)")
        print(f"  │  Q[Green NS]  : {q[0]:>8.3f}")
        print(f"  │  Q[Green EW]  : {q[1]:>8.3f}")
        print(f"  │  DECISION     : ★ {actions[best]}  ({conf} confidence)")
        print(f"  └" + "─"*60)

        count += 1
        if count >= limit:
            print(f"\n  ... showing {limit} states. More exist in full Q-table.")
            break

    print(f"\n  Total trained states shown: {count}")
    print("="*75)

# =============================================================================
# PEDESTRIAN
# =============================================================================
def print_pedestrian(qtable, limit=50):
    cfg      = TrafficEnv.TASK_CONFIG["pedestrian"]
    bins     = cfg["state_bins"]
    n_states = int(np.prod(bins))
    actions  = cfg["action_meanings"]

    print("\n" + "="*75)
    print("  PEDESTRIAN Q-TABLE — Pedestrian Crossing Control (FIXED v2)")
    print("  State variables: waiting_pedestrians, waiting_vehicles,")
    print("                   ped_max_wait, veh_max_wait, current_phase, phase_elapsed")
    print("  Actions: [0] Allow Pedestrians   [1] Allow Vehicles")
    print("  Safety : If pedestrian waits > 6 steps → FORCED pedestrian phase")
    print("="*75)

    count = 0
    for s in range(int(np.prod(bins))):
        q = qtable[s]
        if np.all(np.abs(q - 1.0) < 0.01): continue

        peds, vehs, ped_wait, veh_wait, phase, elapsed = decode_state(s, bins)
        best = int(np.argmax(q))

        safety = "🚨 SAFETY OVERRIDE ZONE" if ped_wait >= 6 else \
                 "⚠ APPROACHING LIMIT"    if ped_wait >= 4 else "✅ Safe"

        if peds == 0 and vehs == 0: sit = "Empty crossing"
        elif peds == 0:             sit = f"No pedestrians — {vehs} vehicle(s) waiting"
        elif vehs == 0:             sit = f"{peds} pedestrian(s) — no vehicles"
        elif ped_wait > veh_wait:   sit = f"Pedestrians more urgent (waited {ped_wait} vs {veh_wait} steps)"
        elif veh_wait > ped_wait:   sit = f"Vehicles more urgent (waited {veh_wait} vs {ped_wait} steps)"
        else:                       sit = f"Equal urgency ({peds} peds, {vehs} vehs)"

        phase_str = "Pedestrian phase" if phase == 0 else "Vehicle phase"
        diff = abs(q[0] - q[1])
        conf = "HIGH" if diff > 5 else "MODERATE" if diff > 2 else "LOW"

        print(f"  ┌─ Pedestrians={peds}, Vehicles={vehs} | {safety}")
        print(f"  │  Situation    : {sit}")
        print(f"  │  Wait times   : Peds waited {ped_wait} steps | Vehs waited {veh_wait} steps")
        print(f"  │  Current phase: {phase_str} (active {elapsed} steps)")
        print(f"  │  Q[Allow Peds]: {q[0]:>8.3f}")
        print(f"  │  Q[Allow Vehs]: {q[1]:>8.3f}")
        print(f"  │  DECISION     : ★ {actions[best]}  ({conf} confidence)")
        print(f"  └" + "─"*60)

        count += 1
        if count >= limit: break

    print(f"\n  Total trained states shown: {count}")
    print("="*75)

# =============================================================================
# PARKING
# =============================================================================
def print_parking(qtable, limit=50):
    cfg      = TrafficEnv.TASK_CONFIG["parking"]
    bins     = cfg["state_bins"]
    n_states = int(np.prod(bins))
    actions  = cfg["action_meanings"]
    occ_labels = ["<25% full","25-50% full","50-75% full","75-100% full","FULL"]

    print("\n" + "="*75)
    print("  PARKING Q-TABLE — Parking Lot Management (FIXED v2)")
    print("  State variables: available_spots, incoming_vehicles,")
    print("                   queue_wait_time, occupancy_rate")
    print("  Actions: [0] Open Zone A  [1] Open Zone B  [2] Close Entry")
    print("  Safety : If vehicles queue > 7 steps and spots exist → Force Open")
    print("="*75)

    count = 0
    for s in range(n_states):
        q = qtable[s]
        if np.all(np.abs(q - 1.0) < 0.01): continue

        spots, incoming, queue_wait, occupancy = decode_state(s, bins)
        best = int(np.argmax(q))

        occ_str = occ_labels[min(occupancy, 4)]
        if spots == 0:            sit = f"LOT FULL — {incoming} vehicles incoming!"
        elif incoming == 0:       sit = f"No vehicles — {spots} spots free"
        elif incoming > spots:    sit = f"OVERFLOW RISK: {incoming} incoming > {spots} spots"
        elif queue_wait >= 5:     sit = f"Long queue! {incoming} vehicles waited {queue_wait} steps"
        else:                     sit = f"Normal: {spots} spots, {incoming} incoming"

        diff = max(q) - sorted(q)[-2]
        conf = "HIGH" if diff > 5 else "MODERATE" if diff > 2 else "LOW"

        print(f"  ┌─ Spots={spots}, Incoming={incoming} | Occupancy: {occ_str}")
        print(f"  │  Situation      : {sit}")
        print(f"  │  Queue wait     : {queue_wait} steps")
        print(f"  │  Q[Open Zone A] : {q[0]:>8.3f}")
        print(f"  │  Q[Open Zone B] : {q[1]:>8.3f}")
        print(f"  │  Q[Close Entry] : {q[2]:>8.3f}")
        print(f"  │  DECISION       : ★ {actions[best]}  ({conf} confidence)")
        print(f"  └" + "─"*60)

        count += 1
        if count >= limit: break

    print(f"\n  Total trained states shown: {count}")
    print("="*75)

# =============================================================================
# SAVE REPORT
# =============================================================================
def save_report(task):
    qtable = load_qtable(task)
    if qtable is None: return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if task == "intersection": print_intersection(qtable, limit=100)
        elif task == "pedestrian": print_pedestrian(qtable, limit=100)
        elif task == "parking":    print_parking(qtable, limit=100)
    path = os.path.join(QTABLE_DIR, f"traffic_{task}_qtable_explained.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [✓] Saved → {path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  UMORDA — TRAFFIC Q-TABLE VIEWER (Fixed v2)")
    print("  Now includes: wait times, safety zones, phase memory")
    print("="*60)

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
            for task, fn in [("intersection",print_intersection),
                             ("pedestrian",print_pedestrian),
                             ("parking",print_parking)]:
                q = load_qtable(task)
                if q is not None: fn(q)
        elif choice == "5":
            for task in ["intersection","pedestrian","parking"]:
                save_report(task)
            print("\n  All reports saved in qtables/ folder!")
        elif choice == "q":
            print("\n  Goodbye! 👋\n"); break
        else:
            print("  [!] Invalid choice.")

if __name__ == "__main__":
    main()
