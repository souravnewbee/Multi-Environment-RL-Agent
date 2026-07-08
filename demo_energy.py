# =============================================================================
# UMORDA — Energy Domain Interactive Demo
# File: demo_energy.py
# =============================================================================

import sys, os, numpy as np
sys.path.append(os.path.dirname(__file__))
from environments.energy_env import EnergyEnv
from agents.energy_agent import EnergyAgent

QTABLE_DIR   = os.path.join(os.path.dirname(__file__), "qtables")
TIME_LABELS  = ["Morning", "Afternoon", "Evening", "Night"]
PRICE_LABELS = ["Cheap", "Normal", "Expensive"]


def load_agent(task):
    env   = EnergyEnv(task=task)
    agent = EnergyAgent(env.observation_space.n, env.action_space.n, epsilon=0.0)
    path  = os.path.join(QTABLE_DIR, f"energy_{task}_qtable.npy")
    if not os.path.exists(path):
        print(f"\n  [!] Q-Table not found. Run training/train_energy.py first!\n")
        sys.exit(1)
    agent.load_qtable(path)
    return agent, env


# =============================================================================
# SOLAR SCHEDULING DEMO
# =============================================================================
def demo_solar():
    print("\n" + "="*65)
    print("  DEMO: Solar Scheduling — Balcony Solar Panel Power Use")
    print("="*65)
    print("  Inputs needed:")
    print("    solar_output     : Solar power generated (0-9, 0=none, 9=max)")
    print("    home_consumption : Current home power usage (0-9)")
    print("    battery_level    : Battery charge level (0-9)")
    print("    time_of_day      : 0=Morning 1=Afternoon 2=Evening 3=Night")

    agent, env = load_agent("solar_scheduling")
    actions    = EnergyEnv.TASK_CONFIG["solar_scheduling"]["action_meanings"]

    while True:
        print("\n  Enter state (or 'q' to go back):")
        try:
            inp = input("  solar consumption battery time: ").strip()
            if inp.lower() == 'q': break
            v = list(map(int, inp.split()))
            assert len(v) == 4
        except:
            print("  [!] Enter 4 integers. Example: 7 3 4 1")
            continue

        solar, consumption, battery, time = v
        state  = env._encode_state(v)
        env._raw_state = v
        action = agent.best_action(state)
        q_vals = agent.q_table[state]

        surplus = solar - consumption
        diff    = abs(max(q_vals) - sorted(q_vals)[-2])
        conf    = "HIGH" if diff > 3 else "MODERATE" if diff > 1 else "LOW"

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  Solar output    : {solar}/9                               │")
        print(f"  │  Home usage      : {consumption}/9                               │")
        print(f"  │  Battery level   : {battery}/9                               │")
        print(f"  │  Time of day     : {TIME_LABELS[time]:<35}│")
        print(f"  │  Solar surplus   : {surplus:+d} units                          │")
        print(f"  │                                                     │")
        print(f"  │  Q[Use Direct]   : {q_vals[0]:>8.3f}                         │")
        print(f"  │  Q[Store Batt]   : {q_vals[1]:>8.3f}                         │")
        print(f"  │  Q[Buy Grid]     : {q_vals[2]:>8.3f}                         │")
        print(f"  │  DECISION        : ★ {actions[action]:<31}│")
        print(f"  │  Confidence      : {conf:<33}│")
        print(f"  └─────────────────────────────────────────────────────┘")

        # Logic check
        if solar == 0 and battery == 0:
            print(f"  Logic: No solar, no battery → Buy from Grid makes sense ✅")
        elif solar > consumption and battery < 8:
            print(f"  Logic: Solar surplus available → Store or Use Direct makes sense ✅")
        elif solar >= consumption:
            print(f"  Logic: Solar covers consumption → Use Direct makes sense ✅")


# =============================================================================
# BATTERY MANAGEMENT DEMO
# =============================================================================
def demo_battery():
    print("\n" + "="*65)
    print("  DEMO: Battery Management — Smart Charge/Discharge")
    print("="*65)
    print("  Inputs needed:")
    print("    battery_level    : Current battery (0-9, 0=empty, 9=full)")
    print("    solar_output     : Solar generation (0-9)")
    print("    grid_price       : 0=Cheap 1=Normal 2=Expensive")
    print("    home_consumption : Current home usage (0-9)")

    agent, env = load_agent("battery_management")
    actions    = EnergyEnv.TASK_CONFIG["battery_management"]["action_meanings"]

    while True:
        print("\n  Enter state (or 'q' to go back):")
        try:
            inp = input("  battery solar price consumption: ").strip()
            if inp.lower() == 'q': break
            v = list(map(int, inp.split()))
            assert len(v) == 4
        except:
            print("  [!] Enter 4 integers. Example: 8 2 2 5")
            continue

        battery, solar, price, consumption = v
        state  = env._encode_state(v)
        env._raw_state = v
        action = agent.best_action(state)
        q_vals = agent.q_table[state]

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  Battery level   : {battery}/9                               │")
        print(f"  │  Solar output    : {solar}/9                               │")
        print(f"  │  Grid price      : {PRICE_LABELS[price]:<33}│")
        print(f"  │  Home consumption: {consumption}/9                               │")
        print(f"  │                                                     │")
        print(f"  │  Q[Charge]       : {q_vals[0]:>8.3f}                         │")
        print(f"  │  Q[Discharge]    : {q_vals[1]:>8.3f}                         │")
        print(f"  │  Q[Keep Idle]    : {q_vals[2]:>8.3f}                         │")
        print(f"  │  DECISION        : ★ {actions[action]:<31}│")
        print(f"  └─────────────────────────────────────────────────────┘")

        if battery >= 8 and price == 2:
            print(f"  Logic: Full battery + expensive grid → Discharge is smart ✅")
        elif solar > 4 and battery < 5:
            print(f"  Logic: Good solar + low battery → Charge makes sense ✅")
        elif battery <= 1:
            print(f"  Logic: ⚠ Battery nearly empty — careful!")


# =============================================================================
# GRID INTERACTION DEMO
# =============================================================================
def demo_grid():
    print("\n" + "="*65)
    print("  DEMO: Grid Interaction — Buy/Sell Energy Optimization")
    print("="*65)
    print("  Inputs needed:")
    print("    grid_price       : 0=Cheap 1=Normal 2=Expensive")
    print("    solar_surplus    : Extra solar beyond home needs (0-9)")
    print("    battery_level    : Battery charge (0-9)")
    print("    home_consumption : Current home usage (0-9)")

    agent, env = load_agent("grid_interaction")
    actions    = EnergyEnv.TASK_CONFIG["grid_interaction"]["action_meanings"]

    while True:
        print("\n  Enter state (or 'q' to go back):")
        try:
            inp = input("  price surplus battery consumption: ").strip()
            if inp.lower() == 'q': break
            v = list(map(int, inp.split()))
            assert len(v) == 4
        except:
            print("  [!] Enter 4 integers. Example: 2 7 5 3")
            continue

        price, surplus, battery, consumption = v
        state  = env._encode_state(v)
        env._raw_state = v
        action = agent.best_action(state)
        q_vals = agent.q_table[state]
        own    = surplus + battery

        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  Grid price      : {PRICE_LABELS[price]:<33}│")
        print(f"  │  Solar surplus   : {surplus}/9                               │")
        print(f"  │  Battery level   : {battery}/9                               │")
        print(f"  │  Home consumption: {consumption}/9                               │")
        print(f"  │  Own power total : {own} units                             │")
        print(f"  │                                                     │")
        print(f"  │  Q[Buy Grid]     : {q_vals[0]:>8.3f}                         │")
        print(f"  │  Q[Sell Grid]    : {q_vals[1]:>8.3f}                         │")
        print(f"  │  Q[Self-Suff]    : {q_vals[2]:>8.3f}                         │")
        print(f"  │  DECISION        : ★ {actions[action]:<31}│")
        print(f"  └─────────────────────────────────────────────────────┘")

        if surplus > 4 and price == 2:
            print(f"  Logic: High surplus + expensive grid → Sell to Grid is smart ✅")
        elif own >= consumption:
            print(f"  Logic: Own power covers needs → Self-sufficient makes sense ✅")
        elif battery == 0 and surplus == 0:
            print(f"  Logic: No own resources → Buy from Grid is necessary ✅")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*65)
    print("  UMORDA — ENERGY DOMAIN DEMO")
    print("  Balcony Solar Panel (Balkonkraftwerk) Optimization")
    print("="*65)

    while True:
        print("\n  [1] Solar Scheduling")
        print("  [2] Battery Management")
        print("  [3] Grid Interaction")
        print("  [q] Quit")

        choice = input("\n  Your choice: ").strip().lower()
        if choice == "1":   demo_solar()
        elif choice == "2": demo_battery()
        elif choice == "3": demo_grid()
        elif choice == "q": print("\n  Goodbye! 👋\n"); break
        else:               print("  [!] Invalid choice.")

if __name__ == "__main__":
    main()
