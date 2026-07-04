"""
view_qtable_finance.py
======================
Inspect trained Finance Q-tables.
Matches the exact structure of view_qtable_hospital.py.

Run from repo root:
    python view_qtable_finance.py
"""

import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# ── Load helpers ──────────────────────────────────────────────────────────────
def load_qtable(task):
    path = f"qtables/finance_{task}.npy"
    if not os.path.exists(path):
        print(f"\n  Q-table not found: {path}")
        print(f"  Run:  python training/train_finance.py  first.\n")
        return None
    return np.load(path)


def load_metadata(task):
    path = f"qtables/finance_{task}_metadata.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Display helpers ───────────────────────────────────────────────────────────
def print_separator(char="=", width=60):
    print(char * width)


def print_metadata(meta):
    if not meta:
        print("  No metadata found.")
        return
    print(f"  Domain          : {meta.get('domain', 'finance')}")
    print(f"  Task            : {meta['task']}")
    print(f"  Episodes        : {meta['episodes']:,}")
    print(f"  Alpha (lr)      : {meta['alpha']}")
    print(f"  Gamma           : {meta['gamma']}")
    print(f"  Q-table shape   : {meta['q_table_shape']}")
    print(f"  State vars      : {meta['state_vars']}")
    print(f"  Actions         : {meta['actions']}")
    print(f"  Trained mean    : {meta['trained_mean']:+.2f}")
    print(f"  Random mean     : {meta['random_mean']:+.2f}")
    print(f"  Improvement     : {meta['improvement_pct']:+.1f}%")
    print(f"  Training time   : {meta['training_time_s']:.1f}s")
    print(f"  Avg last 5k ep  : {meta['avg_last_5k']:+.2f}")
    print(f"  Avg first 5k ep : {meta['avg_first_5k']:+.2f}")
    if meta.get("tickers_used"):
        print(f"  Tickers trained : {meta['tickers_used']}")
    print(f"  Data source     : {meta.get('data_source', 'simulation')}")


# ── Q-table analysis ──────────────────────────────────────────────────────────
def analyse_qtable(Q, task):
    flat = Q.flatten()
    nonzero = flat[flat != 0]
    print(f"\n  Q-table stats:")
    print(f"    Shape         : {Q.shape}")
    print(f"    Total cells   : {flat.size:,}")
    print(f"    Non-zero cells: {len(nonzero):,} "
          f"({100*len(nonzero)/flat.size:.1f}% filled)")
    print(f"    Min Q-value   : {flat.min():+.4f}")
    print(f"    Max Q-value   : {flat.max():+.4f}")
    print(f"    Mean Q-value  : {flat.mean():+.4f}")
    if len(nonzero):
        print(f"    Mean (nonzero): {nonzero.mean():+.4f}")


# ── Policy display: show best action per state ────────────────────────────────
def show_policy_trading(Q):
    actions = ["Buy", "Sell", "Hold"]
    trends  = ["CRASH(-2)", "DOWN(-1)", "FLAT(0)", "UP(+1)", "BULL(+2)"]
    cash_labels = ["$0-200", "$200-500", "$500-1k", "$1k-1.5k", "$1.5k-2k", "$2k+"]
    print("\n  Policy Table (Trading) — Best action per state")
    print(f"  {'Trend':<12} {'Shares':>8} {'Cash':>10}  →  Best Action")
    print("  " + "-"*50)
    for t_idx, trend in enumerate(trends):
        for s_idx in range(4):
            for c_idx, cash_lbl in enumerate(cash_labels):
                if t_idx < Q.shape[0] and s_idx < Q.shape[1] and c_idx < Q.shape[2]:
                    best  = int(np.argmax(Q[t_idx, s_idx, c_idx]))
                    qvals = Q[t_idx, s_idx, c_idx]
                    print(f"  {trend:<12} {s_idx*4:>6} shs  {cash_lbl:>10}"
                          f"  →  {actions[best]:<5} "
                          f"(Q: {qvals[0]:+.2f} / {qvals[1]:+.2f} / {qvals[2]:+.2f})")


def show_policy_savings(Q):
    actions = ["Save More", "Spend Normal", "Invest"]
    print("\n  Policy Table (Savings) — Best action per income/savings/months state")
    print(f"  {'Income Bucket':>15} {'Savings Bucket':>15} {'Months Bucket':>14}  →  Best Action")
    print("  " + "-"*65)
    income_labels  = ["Very Low", "Low", "Medium", "High", "Very High", "Max"]
    savings_labels = ["Empty", "Low", "Medium", "High", "Full"]
    months_labels  = ["1-3", "4-6", "7-9", "10-12", "12+"]
    for i in range(Q.shape[0]):
        for s in range(Q.shape[1]):
            for m in range(Q.shape[2]):
                best  = int(np.argmax(Q[i, s, m]))
                qvals = Q[i, s, m]
                il = income_labels[i] if i < len(income_labels) else str(i)
                sl = savings_labels[s] if s < len(savings_labels) else str(s)
                ml = months_labels[m] if m < len(months_labels) else str(m)
                print(f"  {il:>15} {sl:>15} {ml:>14}  →  {actions[best]}")


def show_policy_budget(Q):
    actions = ["Allocate Full", "Allocate Partial", "Defer"]
    print("\n  Policy Table (Budget) — Best action per budget_used/urgent/depts state")
    print(f"  {'Budget Used':>12} {'Urgent Req':>12} {'Depts Left':>12}  →  Best Action")
    print("  " + "-"*55)
    used_labels   = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    urgent_labels = ["0", "1", "2", "3", "4+"]
    depts_labels  = ["0", "1", "2", "3", "4+"]
    for u in range(Q.shape[0]):
        for urg in range(Q.shape[1]):
            for d in range(Q.shape[2]):
                best  = int(np.argmax(Q[u, urg, d]))
                qvals = Q[u, urg, d]
                ul = used_labels[u]   if u   < len(used_labels)   else str(u)
                el = urgent_labels[urg] if urg < len(urgent_labels) else str(urg)
                dl = depts_labels[d]  if d   < len(depts_labels)  else str(d)
                print(f"  {ul:>12} {el:>12} {dl:>12}  →  {actions[best]}")


# ── Main viewer ───────────────────────────────────────────────────────────────
def view_task(task):
    print_separator()
    print(f"  FINANCE Q-TABLE VIEWER — {task.upper()}")
    print_separator()

    Q    = load_qtable(task)
    meta = load_metadata(task)

    if Q is None:
        return

    print_metadata(meta)
    analyse_qtable(Q, task)

    show_policy = input(f"\n  Show full policy table for {task}? (y/n): ").strip().lower()
    if show_policy == "y":
        if task == "trading":
            show_policy_trading(Q)
        elif task == "savings":
            show_policy_savings(Q)
        elif task == "budget":
            show_policy_budget(Q)


def main():
    print("\n" + "="*60)
    print("  UMORDA — Finance Q-Table Viewer")
    print("="*60)

    # Check if any Q-tables exist
    tasks_available = []
    for task in ["trading", "savings", "budget"]:
        if os.path.exists(f"qtables/finance_{task}.npy"):
            tasks_available.append(task)

    if not tasks_available:
        print("\n  No Finance Q-tables found.")
        print("  Run:  python training/train_finance.py  first.\n")
        return

    print(f"\n  Available Q-tables: {tasks_available}")

    # Summary overview first
    print(f"\n{'='*65}")
    print(f"{'FINANCE DOMAIN — QUICK SUMMARY':^65}")
    print(f"{'='*65}")
    print(f"  {'Task':<12} {'Shape':>16} {'Trained':>10} {'Random':>10} {'Improve%':>10}")
    print(f"  {'-'*63}")
    for task in tasks_available:
        Q    = load_qtable(task)
        meta = load_metadata(task)
        if Q is not None and meta:
            print(f"  {task:<12} "
                  f"{str(Q.shape):>16} "
                  f"{meta['trained_mean']:>+10.1f} "
                  f"{meta['random_mean']:>+10.1f} "
                  f"{meta['improvement_pct']:>+9.1f}%")
    print(f"{'='*65}")

    # Per-task detailed view
    while True:
        print("\n  Which task to inspect?")
        for i, task in enumerate(tasks_available):
            print(f"  {i+1}. {task}")
        print(f"  {len(tasks_available)+1}. Exit")

        choice = input("\n  Enter choice: ").strip()

        try:
            idx = int(choice) - 1
            if idx == len(tasks_available):
                print("\n  Exiting viewer. Goodbye!\n")
                break
            elif 0 <= idx < len(tasks_available):
                view_task(tasks_available[idx])
            else:
                print("  Invalid choice.")
        except ValueError:
            print("  Please enter a number.")


if __name__ == "__main__":
    main()
