# =============================================================================
# UMORDA — Traffic Domain Interactive Demo (FIXED v2)
# File: demo_traffic.py
# =============================================================================

import sys, os, numpy as np
sys.path.append(os.path.dirname(__file__))
from environments.traffic_env import TrafficEnv
from agents.traffic_agent import TrafficAgent

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "qtables")

def load_agent(task):
    env   = TrafficEnv(task=task)
    agent = TrafficAgent(env.observation_space.n, env.action_space.n, epsilon=0.0)
    path  = os.path.join(QTABLE_DIR, f"traffic_{task}_qtable.npy")
    if not os.path.exists(path):
        print(f"\n  [!] Q-Table not found. Run training/train_traffic.py first!\n")
        sys.exit(1)
    agent.load_qtable(path)
    return agent, env

# =============================================================================
# INTERSECTION DEMO
# =============================================================================
def demo_intersection():
    print("\n" + "="*65)
    print("  DEMO: Single Traffic Intersection Control (FIXED v2)")
    print("="*65)
    print("  Inputs needed:")
    print("    cars_NS      : Total cars from North+South (0-9)")
    print("    cars_EW      : Total cars from East+West   (0-9)")
    print("    current_phase: Current signal (0=GreenNS, 1=GreenEW)")
    print("    phase_elapsed: Steps signal has been active (0-9)")
    print("    wait_NS      : How long NS has been waiting (0-9)")
    print("    wait_EW      : How long EW has been waiting (0-9)")
    print("\n  Safety rule: If any direction waits ≥ 8 steps → FORCED green!")

    agent, env = load_agent("intersection")
    actions    = TrafficEnv.TASK_CONFIG["intersection"]["action_meanings"]
    limit      = TrafficEnv.TASK_CONFIG["intersection"]["max_wait_limit"]

    while True:
        print("\n  Enter state (or 'q' to go back):")
        try:
            inp = input("  cars_NS cars_EW phase elapsed wait_NS wait_EW: ").strip()
            if inp.lower() == 'q': break
            v = list(map(int, inp.split()))
            assert len(v) == 6
        except:
            print("  [!] Enter 6 integers. Example: 5 2 0 3 4 1")
            continue

        cars_NS, cars_EW, phase, elapsed, wait_NS, wait_EW = v
        state          = env._encode_state(v)
        env._raw_state = v   # set raw state so safety override can read it
        forced         = env._safety_override(agent.best_action(state))
        action         = forced
        q_vals = agent.q_table[state]

        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  NS: {cars_NS} cars (waited {wait_NS} steps)                    │")
        print(f"  │  EW: {cars_EW} cars (waited {wait_EW} steps)                    │")
        print(f"  │  Current phase: {'GreenNS' if phase==0 else 'GreenEW'} (active {elapsed} steps)       │")
        print(f"  │                                                 │")
        print(f"  │  Q[Green NS]: {q_vals[0]:>8.3f}                          │")
        print(f"  │  Q[Green EW]: {q_vals[1]:>8.3f}                          │")

        if wait_NS >= limit or wait_EW >= limit:
            print(f"  │  ⚠ SAFETY OVERRIDE TRIGGERED!                   │")
        print(f"  │  → DECISION: ★ {actions[action]:<33}│")
        print(f"  └─────────────────────────────────────────────────┘")

        # Logic check
        if wait_NS >= limit:
            print(f"  Safety check: 🚨 NS waited {wait_NS} steps — FORCED Green NS!")
        elif wait_EW >= limit:
            print(f"  Safety check: 🚨 EW waited {wait_EW} steps — FORCED Green EW!")
        elif wait_NS > wait_EW:
            exp = "Green NS"
            print(f"  Logic check : NS waited longer ({wait_NS} vs {wait_EW}) → {'✅' if exp in actions[action] else '⚠'} {actions[action]}")
        elif wait_EW > wait_NS:
            exp = "Green EW"
            print(f"  Logic check : EW waited longer ({wait_EW} vs {wait_NS}) → {'✅' if exp in actions[action] else '⚠'} {actions[action]}")
        else:
            print(f"  Logic check : Equal wait — agent uses car count to decide ✅")

# =============================================================================
# PEDESTRIAN DEMO
# =============================================================================
def demo_pedestrian():
    print("\n" + "="*65)
    print("  DEMO: Pedestrian Crossing Control (FIXED v2)")
    print("="*65)
    print("  Inputs needed:")
    print("    peds         : Waiting pedestrians (0-9)")
    print("    vehs         : Waiting vehicles    (0-9)")
    print("    ped_wait     : How long peds waited (0-9)")
    print("    veh_wait     : How long vehs waited (0-9)")
    print("    phase        : Current phase (0=PedPhase, 1=VehPhase)")
    print("    elapsed      : Steps in current phase (0-9)")
    print("\n  Safety rule: If pedestrian waits ≥ 6 steps → FORCED ped phase!")

    agent, env = load_agent("pedestrian")
    actions    = TrafficEnv.TASK_CONFIG["pedestrian"]["action_meanings"]
    ped_limit  = TrafficEnv.TASK_CONFIG["pedestrian"]["max_ped_wait"]

    while True:
        print("\n  Enter state (or 'q' to go back):")
        try:
            inp = input("  peds vehs ped_wait veh_wait phase elapsed: ").strip()
            if inp.lower() == 'q': break
            v = list(map(int, inp.split()))
            assert len(v) == 6
        except:
            print("  [!] Enter 6 integers. Example: 4 2 5 1 1 3")
            continue

        peds, vehs, ped_wait, veh_wait, phase, elapsed = v
        state          = env._encode_state(v)
        env._raw_state = v
        action         = env._safety_override(agent.best_action(state))
        q_vals         = agent.q_table[state]

        safety_status = "🚨 SAFETY OVERRIDE!" if ped_wait >= ped_limit else \
                        "⚠ Approaching limit" if ped_wait >= ped_limit-2 else "✅ Safe"

        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  Pedestrians: {peds} (waited {ped_wait} steps) {safety_status:<15}│")
        print(f"  │  Vehicles   : {vehs} (waited {veh_wait} steps)                  │")
        print(f"  │  Phase      : {'PedPhase' if phase==0 else 'VehPhase'} (active {elapsed} steps)          │")
        print(f"  │                                                 │")
        print(f"  │  Q[Allow Peds]: {q_vals[0]:>8.3f}                        │")
        print(f"  │  Q[Allow Vehs]: {q_vals[1]:>8.3f}                        │")
        print(f"  │  → DECISION  : ★ {actions[action]:<31}│")
        print(f"  └─────────────────────────────────────────────────┘")

# =============================================================================
# PARKING DEMO
# =============================================================================
def demo_parking():
    print("\n" + "="*65)
    print("  DEMO: Parking Lot Management (FIXED v2)")
    print("="*65)
    print("  Inputs needed:")
    print("    spots      : Available spots (0-19)")
    print("    incoming   : Incoming vehicles (0-9)")
    print("    queue_wait : How long queue has waited (0-9)")
    print("    occupancy  : 0=<25% 1=25-50% 2=50-75% 3=75-100% 4=FULL")

    agent, env = load_agent("parking")
    actions    = TrafficEnv.TASK_CONFIG["parking"]["action_meanings"]
    occ_labels = ["<25% full","25-50% full","50-75% full","75-100% full","FULL"]

    while True:
        print("\n  Enter state (or 'q' to go back):")
        try:
            inp = input("  spots incoming queue_wait occupancy: ").strip()
            if inp.lower() == 'q': break
            v = list(map(int, inp.split()))
            assert len(v) == 4
        except:
            print("  [!] Enter 4 integers. Example: 3 7 6 3")
            continue

        spots, incoming, queue_wait, occupancy = v
        state          = env._encode_state(v)
        env._raw_state = v
        action         = env._safety_override(agent.best_action(state))
        q_vals         = agent.q_table[state]

        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  Available spots : {spots:<30}│")
        print(f"  │  Incoming vehs   : {incoming:<30}│")
        print(f"  │  Queue wait time : {queue_wait} steps{'':<24}│")
        print(f"  │  Occupancy       : {occ_labels[min(occupancy,4)]:<30}│")
        print(f"  │                                                 │")
        print(f"  │  Q[Open Zone A]  : {q_vals[0]:>8.3f}                      │")
        print(f"  │  Q[Open Zone B]  : {q_vals[1]:>8.3f}                      │")
        print(f"  │  Q[Close Entry]  : {q_vals[2]:>8.3f}                      │")
        print(f"  │  → DECISION      : ★ {actions[action]:<27}│")
        print(f"  └─────────────────────────────────────────────────┘")

# =============================================================================
# MAIN MENU
# =============================================================================
def main():
    print("\n" + "="*65)
    print("  UMORDA — TRAFFIC DOMAIN INTERACTIVE DEMO (FIXED v2)")
    print("  Features: Wait time awareness, Hard safety guarantees,")
    print("            Phase memory, Wider state space")
    print("="*65)

    while True:
        print("\n  Select task:")
        print("  [1] Intersection Control")
        print("  [2] Pedestrian Crossing")
        print("  [3] Parking Management")
        print("  [q] Quit")

        choice = input("\n  Your choice: ").strip().lower()
        if choice == "1":   demo_intersection()
        elif choice == "2": demo_pedestrian()
        elif choice == "3": demo_parking()
        elif choice == "q": print("\n  Goodbye! 👋\n"); break
        else:               print("  [!] Invalid choice.")

if __name__ == "__main__":
    main()
