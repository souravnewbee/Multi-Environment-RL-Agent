# =============================================================================
# UMORDA — Traffic Domain Interactive Demo
# File: demo_traffic.py
# Usage: python demo_traffic.py
# =============================================================================

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))

from environments.traffic_env import TrafficEnv
from agents.traffic_agent import TrafficAgent

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "qtables")


# =============================================================================
# Helper — load agent
# =============================================================================
def load_agent(task: str) -> TrafficAgent:
    env   = TrafficEnv(task=task)
    agent = TrafficAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        epsilon=0.0,   # No exploration during demo
    )
    path = os.path.join(QTABLE_DIR, f"traffic_{task}_qtable.npy")
    if not os.path.exists(path):
        print(f"\n  [!] Q-Table not found at {path}")
        print(f"      Please run:  python training/train_traffic.py  first!\n")
        sys.exit(1)
    agent.load_qtable(path)
    return agent, env


# =============================================================================
# Demo — Intersection
# =============================================================================
def demo_intersection():
    print("\n" + "="*60)
    print("  DEMO: Single Traffic Intersection Control")
    print("="*60)
    print("  State variables: cars_N, cars_S, cars_E, cars_W (0-4)")
    print("  Actions: 0=Green NS | 1=Green EW")

    agent, env = load_agent("intersection")
    cfg = TrafficEnv.TASK_CONFIG["intersection"]
    agent.print_qtable(cfg["action_meanings"], title="INTERSECTION Q-TABLE (Top Rows)")

    while True:
        print("\n  Enter traffic state (or 'q' to go back):")
        try:
            inp = input("  cars_N cars_S cars_E cars_W (e.g. 3 1 2 4): ").strip()
            if inp.lower() == 'q':
                break
            vals = list(map(int, inp.split()))
            assert len(vals) == 4
            assert all(0 <= v <= 4 for v in vals)
        except Exception:
            print("  [!] Please enter 4 integers between 0 and 4.")
            continue

        raw_state = vals
        state = env._encode_state(raw_state)
        action = agent.best_action(state)
        q_vals = agent.q_table[state]

        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  State  : N={vals[0]} S={vals[1]} E={vals[2]} W={vals[3]}              │")
        print(f"  │  Q-Values: GreenNS={q_vals[0]:+.3f}  GreenEW={q_vals[1]:+.3f}  │")
        print(f"  │  → BEST ACTION: {cfg['action_meanings'][action]:<26}│")
        print(f"  └─────────────────────────────────────────┘")

        ns = vals[0] + vals[1]
        ew = vals[2] + vals[3]
        expected = "Green NS" if ns >= ew else "Green EW"
        chosen   = cfg["action_meanings"][action]
        match = "✅ CORRECT" if expected.lower() in chosen.lower() else "⚠️  CHECK"
        print(f"  Logic check: NS={ns} cars vs EW={ew} cars → {match}")


# =============================================================================
# Demo — Pedestrian
# =============================================================================
def demo_pedestrian():
    print("\n" + "="*60)
    print("  DEMO: Pedestrian Crossing Control")
    print("="*60)
    print("  State variables: waiting_pedestrians (0-5), waiting_vehicles (0-5)")
    print("  Actions: 0=Allow Pedestrians | 1=Allow Vehicles")

    agent, env = load_agent("pedestrian")
    cfg = TrafficEnv.TASK_CONFIG["pedestrian"]
    agent.print_qtable(cfg["action_meanings"], title="PEDESTRIAN Q-TABLE (Top Rows)")

    while True:
        print("\n  Enter pedestrian crossing state (or 'q' to go back):")
        try:
            inp = input("  waiting_pedestrians waiting_vehicles (e.g. 4 1): ").strip()
            if inp.lower() == 'q':
                break
            vals = list(map(int, inp.split()))
            assert len(vals) == 2
            assert all(0 <= v <= 5 for v in vals)
        except Exception:
            print("  [!] Please enter 2 integers between 0 and 5.")
            continue

        raw_state = vals
        state = env._encode_state(raw_state)
        action = agent.best_action(state)
        q_vals = agent.q_table[state]

        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Pedestrians waiting : {vals[0]:<19}│")
        print(f"  │  Vehicles waiting    : {vals[1]:<19}│")
        print(f"  │  Q-Values: Peds={q_vals[0]:+.3f}  Vehs={q_vals[1]:+.3f}     │")
        print(f"  │  → BEST ACTION: {cfg['action_meanings'][action]:<26}│")
        print(f"  └─────────────────────────────────────────┘")

        if vals[0] > 3:
            safety = "⚠️  HIGH pedestrian wait — safety priority!"
        elif vals[0] == 0:
            safety = "ℹ️  No pedestrians waiting"
        else:
            safety = "✅ Normal situation"
        print(f"  Safety check: {safety}")


# =============================================================================
# Demo — Parking
# =============================================================================
def demo_parking():
    print("\n" + "="*60)
    print("  DEMO: Parking Lot Management")
    print("="*60)
    print("  State variables: available_spots (0-10), incoming_vehicles (0-5)")
    print("  Actions: 0=Open Zone A | 1=Open Zone B | 2=Close Entry")

    agent, env = load_agent("parking")
    cfg = TrafficEnv.TASK_CONFIG["parking"]
    agent.print_qtable(cfg["action_meanings"], title="PARKING Q-TABLE (Top Rows)")

    while True:
        print("\n  Enter parking state (or 'q' to go back):")
        try:
            inp = input("  available_spots incoming_vehicles (e.g. 3 5): ").strip()
            if inp.lower() == 'q':
                break
            vals = list(map(int, inp.split()))
            assert len(vals) == 2
            assert 0 <= vals[0] <= 10
            assert 0 <= vals[1] <= 5
        except Exception:
            print("  [!] available_spots: 0-10, incoming_vehicles: 0-5")
            continue

        raw_state = vals
        state = env._encode_state(raw_state)
        action = agent.best_action(state)
        q_vals = agent.q_table[state]

        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Available spots     : {vals[0]:<19}│")
        print(f"  │  Incoming vehicles   : {vals[1]:<19}│")
        print(f"  │  Q: ZoneA={q_vals[0]:+.2f} ZoneB={q_vals[1]:+.2f} Close={q_vals[2]:+.2f}│")
        print(f"  │  → BEST ACTION: {cfg['action_meanings'][action]:<26}│")
        print(f"  └─────────────────────────────────────────┘")

        if vals[0] == 0:
            logic = "⚠️  Lot FULL — Close Entry makes sense"
        elif vals[1] > vals[0]:
            logic = "⚠️  More cars than spots — manage carefully"
        else:
            logic = "✅ Enough spots for incoming vehicles"
        print(f"  Logic check: {logic}")


# =============================================================================
# MAIN MENU
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  UMORDA — TRAFFIC DOMAIN DEMO")
    print("  Multi-Objective Reinforcement Learning Agent")
    print("="*60)

    menu = {
        "1": ("Single Traffic Intersection Control", demo_intersection),
        "2": ("Pedestrian Crossing Control",         demo_pedestrian),
        "3": ("Parking Lot Management",              demo_parking),
        "q": ("Quit",                                None),
    }

    while True:
        print("\n  Select a task to demo:")
        for key, (label, _) in menu.items():
            print(f"    [{key}] {label}")

        choice = input("\n  Your choice: ").strip().lower()

        if choice == 'q':
            print("\n  Goodbye! 👋\n")
            break
        elif choice in menu and menu[choice][1]:
            menu[choice][1]()
        else:
            print("  [!] Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
