# =============================================================================
# UMORDA — Energy Domain: Human-Readable Q-Table Viewer
# File: view_qtable_energy.py
# =============================================================================

import sys, os, numpy as np, io, contextlib
sys.path.append(os.path.dirname(__file__))
from environments.energy_env import EnergyEnv

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "qtables")

TIME_LABELS  = ["Morning", "Afternoon", "Evening", "Night"]
PRICE_LABELS = ["Cheap", "Normal", "Expensive"]
OCC_LABELS   = ["<25% full", "25-50% full", "50-75% full", "75-100% full", "FULL"]


def load_qtable(task):
    path = os.path.join(QTABLE_DIR, f"energy_{task}_qtable.npy")
    if not os.path.exists(path):
        print(f"  [!] Q-Table not found. Run: python training/train_energy.py first!")
        return None
    return np.load(path)


def decode_state(state_idx, bins):
    values = []
    for b in reversed(bins):
        values.append(state_idx % b)
        state_idx //= b
    return list(reversed(values))


# =============================================================================
# SOLAR SCHEDULING
# =============================================================================
def print_solar_scheduling(qtable, limit=50):
    cfg      = EnergyEnv.TASK_CONFIG["solar_scheduling"]
    bins     = cfg["state_bins"]
    n_states = int(np.prod(bins))
    actions  = cfg["action_meanings"]

    print("\n" + "="*72)
    print("  SOLAR SCHEDULING Q-TABLE — Balcony Solar Panel Power Scheduling")
    print("  State: solar_output, home_consumption, battery_level, time_of_day")
    print("  Actions: [0] Use Solar Directly  [1] Store in Battery  [2] Buy from Grid")
    print("="*72)

    count = 0
    for s in range(n_states):
        q = qtable[s]
        if np.all(q == 0): continue

        solar, consumption, battery, time = decode_state(s, bins)
        best    = int(np.argmax(q))
        surplus = solar - consumption

        if solar == 0:
            situation = f"No solar — {TIME_LABELS[time]}, battery at {battery*11}%"
        elif solar > consumption:
            situation = f"Solar surplus of {surplus} units — {TIME_LABELS[time]}"
        elif solar == consumption:
            situation = f"Solar exactly meets consumption — {TIME_LABELS[time]}"
        else:
            situation = f"Solar deficit ({consumption-solar} units short) — {TIME_LABELS[time]}"

        battery_pct = round(battery / 9 * 100)
        batt_status = "Full" if battery >= 8 else "High" if battery >= 6 else \
                      "Medium" if battery >= 3 else "Low" if battery >= 1 else "Empty"

        diff = max(q) - sorted(q)[-2] if len(q) > 1 else 0
        conf = "HIGH" if diff > 3 else "MODERATE" if diff > 1 else "LOW"

        print(f"  ┌─ Solar={solar}/9  Consumption={consumption}/9  Battery={battery}/9 ({batt_status})  Time={TIME_LABELS[time]}")
        print(f"  │  Situation           : {situation}")
        print(f"  │  Q[Use Solar Direct] : {q[0]:>8.3f}")
        print(f"  │  Q[Store in Battery] : {q[1]:>8.3f}")
        print(f"  │  Q[Buy from Grid]    : {q[2]:>8.3f}")
        print(f"  │  DECISION            : ★ {actions[best]}  ({conf} confidence)")
        print(f"  └" + "─"*62)

        count += 1
        if count >= limit: break

    print(f"\n  Total trained states shown: {count}")
    print("="*72)


# =============================================================================
# BATTERY MANAGEMENT
# =============================================================================
def print_battery_management(qtable, limit=50):
    cfg      = EnergyEnv.TASK_CONFIG["battery_management"]
    bins     = cfg["state_bins"]
    n_states = int(np.prod(bins))
    actions  = cfg["action_meanings"]

    print("\n" + "="*72)
    print("  BATTERY MANAGEMENT Q-TABLE — Charge/Discharge Optimization")
    print("  State: battery_level, solar_output, grid_price, home_consumption")
    print("  Actions: [0] Charge Battery  [1] Discharge Battery  [2] Keep Idle")
    print("="*72)

    count = 0
    for s in range(n_states):
        q = qtable[s]
        if np.all(q == 0): continue

        battery, solar, price, consumption = decode_state(s, bins)
        best = int(np.argmax(q))

        batt_status = "Full" if battery >= 8 else "High" if battery >= 6 else \
                      "Medium" if battery >= 3 else "Low" if battery >= 1 else "Empty"
        price_label = PRICE_LABELS[price]

        if battery >= 8 and price == 2:
            situation = f"Battery full + grid expensive → great time to discharge"
        elif solar > 5 and battery < 5:
            situation = f"Strong solar ({solar}/9) + low battery → charge opportunity"
        elif battery <= 1 and price == 2:
            situation = f"⚠ Battery nearly empty + expensive grid — critical!"
        elif price == 0 and battery < 7:
            situation = f"Cheap grid price — good time to top up battery"
        else:
            situation = f"Battery {batt_status}, grid {price_label}, solar={solar}/9"

        diff = max(q) - sorted(q)[-2] if len(q) > 1 else 0
        conf = "HIGH" if diff > 3 else "MODERATE" if diff > 1 else "LOW"

        print(f"  ┌─ Battery={battery}/9 ({batt_status})  Solar={solar}/9  Grid={price_label}  Consumption={consumption}/9")
        print(f"  │  Situation              : {situation}")
        print(f"  │  Q[Charge Battery]      : {q[0]:>8.3f}")
        print(f"  │  Q[Discharge Battery]   : {q[1]:>8.3f}")
        print(f"  │  Q[Keep Idle]           : {q[2]:>8.3f}")
        print(f"  │  DECISION               : ★ {actions[best]}  ({conf} confidence)")
        print(f"  └" + "─"*62)

        count += 1
        if count >= limit: break

    print(f"\n  Total trained states shown: {count}")
    print("="*72)


# =============================================================================
# GRID INTERACTION
# =============================================================================
def print_grid_interaction(qtable, limit=50):
    cfg      = EnergyEnv.TASK_CONFIG["grid_interaction"]
    bins     = cfg["state_bins"]
    n_states = int(np.prod(bins))
    actions  = cfg["action_meanings"]

    print("\n" + "="*72)
    print("  GRID INTERACTION Q-TABLE — Buy/Sell Grid Energy Optimization")
    print("  State: grid_price, solar_surplus, battery_level, home_consumption")
    print("  Actions: [0] Buy from Grid  [1] Sell to Grid  [2] Stay Self-Sufficient")
    print("="*72)

    count = 0
    for s in range(n_states):
        q = qtable[s]
        if np.all(q == 0): continue

        price, surplus, battery, consumption = decode_state(s, bins)
        best        = int(np.argmax(q))
        price_label = PRICE_LABELS[price]
        own_power   = surplus + battery

        if surplus > 5 and price == 2:
            situation = f"High surplus ({surplus}/9) + expensive grid → sell for profit!"
        elif battery == 0 and surplus == 0:
            situation = f"No solar/battery — must buy from grid"
        elif own_power >= consumption:
            situation = f"Self-sufficient! Own power ({own_power}) covers consumption ({consumption})"
        elif price == 0 and battery < 5:
            situation = f"Grid is cheap — good time to buy and store"
        else:
            situation = f"Grid={price_label}, Surplus={surplus}/9, Battery={battery}/9"

        diff = max(q) - sorted(q)[-2] if len(q) > 1 else 0
        conf = "HIGH" if diff > 3 else "MODERATE" if diff > 1 else "LOW"

        print(f"  ┌─ Grid={price_label}  Surplus={surplus}/9  Battery={battery}/9  Consumption={consumption}/9")
        print(f"  │  Situation                : {situation}")
        print(f"  │  Q[Buy from Grid]         : {q[0]:>8.3f}")
        print(f"  │  Q[Sell to Grid]          : {q[1]:>8.3f}")
        print(f"  │  Q[Stay Self-Sufficient]  : {q[2]:>8.3f}")
        print(f"  │  DECISION                 : ★ {actions[best]}  ({conf} confidence)")
        print(f"  └" + "─"*62)

        count += 1
        if count >= limit: break

    print(f"\n  Total trained states shown: {count}")
    print("="*72)


# =============================================================================
# SAVE REPORT
# =============================================================================
def save_report(task):
    qtable = load_qtable(task)
    if qtable is None: return
    buf = io.StringIO()
    fn_map = {
        "solar_scheduling":   print_solar_scheduling,
        "battery_management": print_battery_management,
        "grid_interaction":   print_grid_interaction,
    }
    with contextlib.redirect_stdout(buf):
        fn_map[task](qtable, limit=200)
    path = os.path.join(QTABLE_DIR, f"energy_{task}_qtable_explained.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [✓] Saved → {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  UMORDA — ENERGY Q-TABLE VIEWER")
    print("  Balcony Solar Panel (Balkonkraftwerk) Optimization")
    print("="*60)

    fn_map = {
        "solar_scheduling":   print_solar_scheduling,
        "battery_management": print_battery_management,
        "grid_interaction":   print_grid_interaction,
    }

    while True:
        print("\n  [1] Solar Scheduling Q-Table")
        print("  [2] Battery Management Q-Table")
        print("  [3] Grid Interaction Q-Table")
        print("  [4] View ALL")
        print("  [5] Save ALL as .txt reports")
        print("  [q] Quit")

        choice = input("\n  Your choice: ").strip().lower()
        if choice == "1":
            q = load_qtable("solar_scheduling")
            if q is not None: print_solar_scheduling(q)
        elif choice == "2":
            q = load_qtable("battery_management")
            if q is not None: print_battery_management(q)
        elif choice == "3":
            q = load_qtable("grid_interaction")
            if q is not None: print_grid_interaction(q)
        elif choice == "4":
            for task, fn in fn_map.items():
                q = load_qtable(task)
                if q is not None: fn(q)
        elif choice == "5":
            for task in fn_map:
                save_report(task)
            print("\n  All reports saved in qtables/ folder!")
        elif choice == "q":
            print("\n  Goodbye! 👋\n"); break
        else:
            print("  [!] Invalid choice.")

if __name__ == "__main__":
    main()
