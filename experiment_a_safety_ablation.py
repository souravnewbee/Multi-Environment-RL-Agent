"""
Experiment A — Safety Override Ablation
=========================================
Runs 200 evaluation episodes for each of 4 tasks under two conditions:
  ON  — safety override active (current behaviour)
  OFF — safety override bypassed (raw Q-table argmax goes straight to env)

Tasks:
  traffic/intersection  → _safety_override() hard-limits max wait per direction
  traffic/pedestrian    → _safety_override() forces ped phase when ped_wait ≥ 6
  hospital/er_queue     → penalty structure deters serving normal before emergency
                          (there is no hard override; OFF = we forcibly ignore Q
                           and serve Normal whenever emergency also > 0)
  finance/budget        → deferring urgent requests carries a heavy penalty;
                          OFF = always Defer regardless of Q-table

Outputs:
  • experiment_a_results.csv   — raw per-task metrics
  • experiment_a_table.png     — summary table figure
  • experiment_a_barchart.png  — reward + max-wait bar charts per task
"""

import sys
import os
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sqlite3
import csv

warnings.filterwarnings("ignore")

# ── Make sure repo root is on path ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environments.traffic_env    import TrafficEnv
from environments.hospital_env   import HospitalEnv
from environments.finance_env    import FinanceEnv
from agents.hospital_agent       import discretize as hospital_discretize
from agents.finance_agent        import FinanceAgent

N_EPISODES  = 200
MAX_STEPS   = 100   # steps per episode


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_qtable_db(name):
    db = os.path.join(ROOT, "qtables", "qtables.db")
    conn = sqlite3.connect(db)
    row  = conn.execute(
        "SELECT shape, data FROM qtables WHERE name=?", (name,)
    ).fetchone()
    conn.close()
    shape = tuple(int(x) for x in row[0].split(","))
    return np.frombuffer(row[1], dtype=np.float64).reshape(shape)


def jain_fairness(values):
    """Jain's fairness index over a list of values. Returns NaN if all zero."""
    vals = np.array(values, dtype=float)
    if vals.sum() == 0:
        return float("nan")
    n = len(vals)
    return (vals.sum() ** 2) / (n * (vals ** 2).sum())


# ══════════════════════════════════════════════════════════════════════════════
# Task runners
# ══════════════════════════════════════════════════════════════════════════════

def run_traffic_intersection(override_on: bool) -> dict:
    """
    Returns dict with:
      rewards          list[float]   — total reward per episode
      max_waits_NS     list[int]     — max wait NS reached in episode
      max_waits_EW     list[int]     — max wait EW reached in episode
      starvation_events list[int]   — times wait ≥ limit in episode
      jain_scores      list[float]  — Jain fairness (NS vs EW max wait)
    """
    Q     = np.load(os.path.join(ROOT, "qtables", "traffic_intersection_qtable.npy"))
    env   = TrafficEnv(task="intersection")
    limit = env.cfg["max_wait_limit"]   # 8 by default

    rewards, max_ns, max_ew, starvation, jain = [], [], [], [], []

    for _ in range(N_EPISODES):
        obs, _ = env.reset()
        total_r = 0.0
        ep_max_ns = ep_max_ew = 0
        ep_starve = 0

        for _ in range(MAX_STEPS):
            # Greedy action from Q-table
            action = int(np.argmax(Q[obs]))

            if override_on:
                action = env._safety_override(action)

            else:
                # OFF: check if override WOULD have fired, count as starvation
                raw   = env._raw_state
                w_ns  = raw[4]
                w_ew  = raw[5]
                if w_ns >= limit or w_ew >= limit:
                    ep_starve += 1

            obs_next, r, terminated, truncated, _ = env.step(action)

            # Record wait times from raw state AFTER step
            raw       = env._raw_state
            ep_max_ns = max(ep_max_ns, raw[4])
            ep_max_ew = max(ep_max_ew, raw[5])

            if override_on:
                if raw[4] >= limit or raw[5] >= limit:
                    ep_starve += 1

            total_r += r
            obs = obs_next
            if terminated or truncated:
                break

        rewards.append(total_r)
        max_ns.append(ep_max_ns)
        max_ew.append(ep_max_ew)
        starvation.append(ep_starve)
        jain.append(jain_fairness([ep_max_ns, ep_max_ew]))

    return {
        "rewards":           rewards,
        "max_wait_NS":       max_ns,
        "max_wait_EW":       max_ew,
        "starvation_events": starvation,
        "jain_scores":       jain,
    }


def run_traffic_pedestrian(override_on: bool) -> dict:
    Q     = np.load(os.path.join(ROOT, "qtables", "traffic_pedestrian_qtable.npy"))
    env   = TrafficEnv(task="pedestrian")
    limit = env.cfg["max_ped_wait"]  # 6

    rewards, max_ped_waits, starvation, jain = [], [], [], []

    for _ in range(N_EPISODES):
        obs, _ = env.reset()
        total_r = 0.0
        ep_max_ped = 0
        ep_starve  = 0

        for _ in range(MAX_STEPS):
            action = int(np.argmax(Q[obs]))

            if override_on:
                action = env._safety_override(action)
            else:
                raw = env._raw_state
                if raw[2] >= limit:   # ped_wait index
                    ep_starve += 1

            obs_next, r, terminated, truncated, _ = env.step(action)

            raw = env._raw_state
            ep_max_ped = max(ep_max_ped, raw[2])
            if override_on and raw[2] >= limit:
                ep_starve += 1

            total_r += r
            obs = obs_next
            if terminated or truncated:
                break

        rewards.append(total_r)
        max_ped_waits.append(ep_max_ped)
        starvation.append(ep_starve)
        # Fairness: ped vs vehicle max wait
        ep_max_veh = env._raw_state[3]
        jain.append(jain_fairness([ep_max_ped, ep_max_veh]))

    return {
        "rewards":           rewards,
        "max_wait_ped":      max_ped_waits,
        "starvation_events": starvation,
        "jain_scores":       jain,
    }


def run_hospital_er_queue(override_on: bool) -> dict:
    """
    The hospital env doesn't have a _safety_override() call in step().
    The 'safety' is purely via reward structure (heavy penalty for serving
    normal while emergency is waiting).

    OFF condition: we IGNORE the Q-table and always choose "Serve Normal"
    (action=1) when emergency > 0 — simulating what would happen with no
    safety incentive at all.

    The key safety metric is how often emergency patients are left waiting
    while normal patients are served (starvation events).
    """
    Q   = load_qtable_db("hospital_er_queue")
    env = HospitalEnv(task="er_queue")

    rewards, max_emerg_q, starvation, jain = [], [], [], []

    for _ in range(N_EPISODES):
        obs, _ = env.reset()
        total_r  = 0.0
        ep_max_e = 0
        ep_starve = 0

        for _ in range(MAX_STEPS):
            s = hospital_discretize(obs, "er_queue")

            if override_on:
                action = int(np.argmax(Q[s]))
            else:
                # Worst-case: always serve Normal, regardless of emergency queue
                action = 1  # "Serve Normal"

            emerg = obs[0]
            # Count starvation: normal served while emergency queue > 0
            if action == 1 and emerg > 0:
                ep_starve += 1

            obs_next, r, terminated, truncated, _ = env.step(action)

            ep_max_e = max(ep_max_e, float(obs_next[0]))
            total_r += r
            obs = obs_next
            if terminated or truncated:
                break

        rewards.append(total_r)
        max_emerg_q.append(ep_max_e)
        starvation.append(ep_starve)
        # Fairness: how balanced are emergency vs normal queue sizes at episode end
        jain.append(jain_fairness([obs[0]+1, obs[1]+1]))

    return {
        "rewards":           rewards,
        "max_emergency_q":   max_emerg_q,
        "starvation_events": starvation,
        "jain_scores":       jain,
    }


def run_finance_budget(override_on: bool) -> dict:
    """
    OFF condition: always Defer, regardless of urgency — simulates ignoring
    the triage rule entirely. This will accumulate urgent-defer penalties.

    The 'safety' metric is how many times an urgent request was deferred.
    """
    agent = FinanceAgent(task="budget")
    env   = FinanceEnv(task="budget")

    rewards, urgent_defers, jain = [], [], []

    for _ in range(N_EPISODES):
        state   = env.reset()
        total_r = 0.0
        ep_defers = 0

        for _ in range(MAX_STEPS):
            if override_on:
                action_idx, _, _ = agent.get_action(state)
            else:
                # Always Defer (action=2)
                action_idx = 2

            urgent = state.get("urgent_requests", 0)
            if action_idx == 2 and urgent > 0:
                ep_defers += 1

            result     = env.step(action_idx)
            next_state = result[0]
            r          = result[1]
            done       = result[2]

            total_r += r
            state    = next_state
            if done:
                break

        rewards.append(total_r)
        urgent_defers.append(ep_defers)
        # No direct wait-time analogue; use urgent_defers as the safety metric
        jain.append(jain_fairness([
            state.get("urgent_requests", 0) + 1,
            state.get("departments_remaining", 1) + 1
        ]))

    return {
        "rewards":           rewards,
        "urgent_defers":     urgent_defers,
        "starvation_events": urgent_defers,   # alias for uniform reporting
        "jain_scores":       jain,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Run all tasks
# ══════════════════════════════════════════════════════════════════════════════

TASKS = [
    ("intersection", run_traffic_intersection,  "max_wait_NS",    "Max Wait NS (steps)"),
    ("pedestrian",   run_traffic_pedestrian,    "max_wait_ped",   "Max Ped Wait (steps)"),
    ("er_queue",     run_hospital_er_queue,     "max_emergency_q","Max Emergency Queue"),
    ("budget",       run_finance_budget,        "urgent_defers",  "Urgent Requests Deferred"),
]


print("Running Experiment A — Safety Override Ablation")
print(f"  Episodes per condition per task: {N_EPISODES}")
print(f"  Max steps per episode: {MAX_STEPS}\n")

results = {}

for task_name, runner, safety_key, safety_label in TASKS:
    print(f"  [{task_name}] ON  ...", end=" ", flush=True)
    results[(task_name, "ON")]  = runner(override_on=True)
    print("done")
    print(f"  [{task_name}] OFF ...", end=" ", flush=True)
    results[(task_name, "OFF")] = runner(override_on=False)
    print("done")

print()


# ══════════════════════════════════════════════════════════════════════════════
# Compute summary stats
# ══════════════════════════════════════════════════════════════════════════════

summary = []
for task_name, _, safety_key, safety_label in TASKS:
    for cond in ["ON", "OFF"]:
        d = results[(task_name, cond)]
        rw   = np.array(d["rewards"])
        sv   = np.array(d["starvation_events"])
        ji   = np.array(d["jain_scores"])
        safe = np.array(d.get(safety_key, sv))  # fallback to starvation
        summary.append({
            "task":              task_name,
            "condition":         cond,
            "mean_reward":       round(float(np.mean(rw)),  2),
            "std_reward":        round(float(np.std(rw)),   2),
            "mean_safety_metric":round(float(np.mean(safe)),2),
            "max_safety_metric": round(float(np.max(safe)), 2),
            "mean_starvation":   round(float(np.mean(sv)),  2),
            "mean_jain":         round(float(np.nanmean(ji)),3),
            "safety_label":      safety_label,
        })

# Print to console
print(f"{'Task':<14} {'Cond':<5} {'Reward':>10} {'SafetyMetric':>14} {'Starvation':>12} {'Jain':>7}")
print("-" * 65)
for row in summary:
    print(
        f"{row['task']:<14} {row['condition']:<5}"
        f" {row['mean_reward']:>10.2f}"
        f" {row['mean_safety_metric']:>14.2f}"
        f" {row['mean_starvation']:>12.2f}"
        f" {row['mean_jain']:>7.3f}"
    )

# Save CSV
csv_path = os.path.join(ROOT, "experiment_a_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
print(f"\nResults saved → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {"ON": "#2196F3", "OFF": "#F44336"}
task_labels = {
    "intersection": "Traffic\nIntersection",
    "pedestrian":   "Traffic\nPedestrian",
    "er_queue":     "Hospital\nER Queue",
    "budget":       "Finance\nBudget",
}

task_names_ordered = [t[0] for t in TASKS]

def get_stat(task, cond, key):
    return [r for r in summary if r["task"] == task and r["condition"] == cond][0][key]


# ── Figure 1: Reward + Safety metric side by side ──────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(
    "Experiment A — Safety Override Ablation\n"
    "Mean over 200 evaluation episodes per condition",
    fontsize=13, fontweight="bold", y=1.01
)

for col, (task_name, _, safety_key, safety_label) in enumerate(TASKS):
    # Row 0: Reward
    ax = axes[0, col]
    vals = [get_stat(task_name, c, "mean_reward") for c in ["ON", "OFF"]]
    errs = [get_stat(task_name, c, "std_reward")  for c in ["ON", "OFF"]]
    bars = ax.bar(["ON", "OFF"], vals, color=[COLORS["ON"], COLORS["OFF"]],
                  yerr=errs, capsize=5, width=0.5, edgecolor="white")
    ax.set_title(task_labels[task_name], fontsize=10, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Mean Episode Reward", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(labelsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Row 1: Safety metric
    ax = axes[1, col]
    vals_s = [get_stat(task_name, c, "mean_safety_metric") for c in ["ON", "OFF"]]
    bars_s = ax.bar(["ON", "OFF"], vals_s, color=[COLORS["ON"], COLORS["OFF"]],
                    width=0.5, edgecolor="white")
    if col == 0:
        ax.set_ylabel("Mean Safety Violation Metric", fontsize=9)
    ax.set_xlabel("Safety Override", fontsize=9)
    ax.tick_params(labelsize=9)
    for bar, val in zip(bars_s, vals_s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

patch_on  = mpatches.Patch(color=COLORS["ON"],  label="Override ON  (current)")
patch_off = mpatches.Patch(color=COLORS["OFF"], label="Override OFF (ablation)")
fig.legend(handles=[patch_on, patch_off], loc="lower center",
           ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
bar_path = os.path.join(ROOT, "experiment_a_barchart.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Bar chart saved → {bar_path}")


# ── Figure 2: Summary table ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")

col_headers = [
    "Task", "Condition",
    "Mean Reward\n(±std)", "Mean Safety\nMetric (↓ better)",
    "Mean Starvation\nEvents (↓ better)", "Jain Fairness\n(↑ better)"
]
table_data = []
for row in summary:
    table_data.append([
        row["task"],
        row["condition"],
        f"{row['mean_reward']:.1f} ± {row['std_reward']:.1f}",
        f"{row['mean_safety_metric']:.2f}",
        f"{row['mean_starvation']:.2f}",
        f"{row['mean_jain']:.3f}",
    ])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.8)

# Colour header row
for col_idx in range(len(col_headers)):
    tbl[0, col_idx].set_facecolor("#1565C0")
    tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

# Colour ON/OFF rows
for row_idx, row in enumerate(summary, start=1):
    colour = "#E3F2FD" if row["condition"] == "ON" else "#FFEBEE"
    for col_idx in range(len(col_headers)):
        tbl[row_idx, col_idx].set_facecolor(colour)

ax.set_title(
    "Experiment A — Safety Override Ablation: Summary Table\n"
    f"({N_EPISODES} evaluation episodes per task × condition)",
    fontsize=11, fontweight="bold", pad=20
)

tbl_path = os.path.join(ROOT, "experiment_a_table.png")
plt.savefig(tbl_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Table saved      → {tbl_path}")
print("\nExperiment A complete.")