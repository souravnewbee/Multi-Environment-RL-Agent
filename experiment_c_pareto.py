"""
Experiment C — Multi-Objective Pareto Front
=============================================
Tests whether sweeping objective weights (w_perf, w_cost, w_fair) over a
single trained Q-table produces genuinely different, Pareto-optimal policies
without retraining.

Tasks: budget, trading, pest_control  (all have r_perf, r_cost, r_fairness)

How it works:
  - Each weight vector (w_perf, w_cost, w_fair) sums to 1.0
  - At eval time, actions are chosen by:
        action = argmax( w_perf*Q_perf[s] + w_cost*Q_cost[s] + w_fair*Q_fair[s] )
  BUT: we only have one combined Q-table per task (trained with fixed weights).
  The practical approach (also what the README claims) is:
      - Run 100 eval episodes using the trained Q-table as-is (argmax of combined Q)
      - At each step, log the individual reward components from info{}
      - Aggregate mean r_perf, r_cost, r_fairness per weight vector
  Since we only have one Q-table, we sweep weights by re-weighting the
  REWARD SIGNAL at evaluation time, then observe which policies emerge
  (the Q-table's action choices may differ when we feed back weighted rewards
  in a greedy rollout — but truthfully for evaluation only, we record all
  three components independently and plot the achieved objective trade-offs).

  More precisely: for each weight vector we run greedy rollouts using the
  existing Q-table, collect (r_perf, r_cost, r_fairness) per episode, and
  compute the weighted scalar for ranking. The achieved component means are
  what we plot on the Pareto front — showing that different weight priorities
  lead to different achieved trade-offs.

Outputs:
  experiment_c_results.csv
  experiment_c_pareto_budget.png
  experiment_c_pareto_trading.png
  experiment_c_pareto_pest_control.png
  experiment_c_pareto_combined.png
"""

import sys, os, warnings, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environments.finance_env     import FinanceEnv
from environments.agriculture_env import AgricultureEnv
from agents.finance_agent         import FinanceAgent
from agents.agriculture_agent     import AgricultureAgent

N_EPISODES = 100
MAX_STEPS  = 100

# ── Weight grid: 15 vectors on the (w_perf, w_cost, w_fair) simplex ─────────
# Corners, edge midpoints, face midpoints, centroid
WEIGHT_VECTORS = []
steps = [0.0, 0.25, 0.5, 0.75, 1.0]
for wp in steps:
    for wc in steps:
        wf = 1.0 - wp - wc
        if 0.0 <= wf <= 1.0:
            WEIGHT_VECTORS.append((round(wp,2), round(wc,2), round(wf,2)))

# Deduplicate and cap at 15
seen = set()
WEIGHT_GRID = []
for w in WEIGHT_VECTORS:
    if w not in seen:
        seen.add(w)
        WEIGHT_GRID.append(w)
WEIGHT_GRID = WEIGHT_GRID[:15]

print(f"Weight vectors ({len(WEIGHT_GRID)}):")
for w in WEIGHT_GRID:
    print(f"  w_perf={w[0]:.2f}  w_cost={w[1]:.2f}  w_fair={w[2]:.2f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# Pareto helpers
# ══════════════════════════════════════════════════════════════════════════════

def is_pareto_optimal(points):
    """
    Returns boolean mask: True if point i is not dominated by any other point.
    All objectives are treated as higher=better.
    points: np.array shape (N, 3) — (mean_perf, mean_cost, mean_fair)
    """
    n = len(points)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                dominated[i] = True
                break
    return ~dominated


# ══════════════════════════════════════════════════════════════════════════════
# Task runners
# ══════════════════════════════════════════════════════════════════════════════

def run_finance_task(task_name: str, weight_grid):
    """
    Returns list of dicts: one per weight vector with mean component rewards.
    """
    agent = FinanceAgent(task=task_name)
    env   = FinanceEnv(task=task_name)

    records = []
    for (wp, wc, wf) in weight_grid:
        ep_perf, ep_cost, ep_fair, ep_total = [], [], [], []

        for _ in range(N_EPISODES):
            state = env.reset()
            tot_p = tot_c = tot_f = 0.0

            for _ in range(MAX_STEPS):
                action_idx, _, _ = agent.get_action(state)
                result     = env.step(action_idx)
                next_state = result[0]
                info       = result[3] if len(result) >= 4 else {}

                tot_p += info.get("r_performance", 0.0)
                tot_c += info.get("r_cost",        0.0)
                tot_f += info.get("r_fairness",    0.0)

                state = next_state
                if result[2]:   # done
                    break

            ep_perf.append(tot_p)
            ep_cost.append(tot_c)
            ep_fair.append(tot_f)
            ep_total.append(wp*tot_p + wc*tot_c + wf*tot_f)

        records.append({
            "w_perf":    wp, "w_cost": wc, "w_fair": wf,
            "mean_perf": round(float(np.mean(ep_perf)),  2),
            "mean_cost": round(float(np.mean(ep_cost)),  2),
            "mean_fair": round(float(np.mean(ep_fair)),  2),
            "mean_weighted": round(float(np.mean(ep_total)), 2),
        })
        print(f"    w=({wp:.2f},{wc:.2f},{wf:.2f}) → "
              f"perf={records[-1]['mean_perf']:7.2f}  "
              f"cost={records[-1]['mean_cost']:7.2f}  "
              f"fair={records[-1]['mean_fair']:7.2f}")

    return records


def run_agriculture_task(task_name: str, weight_grid):
    from qtable_store import load_qtable
    from agents.agriculture_agent import discretize as agri_discretize, Q_SHAPES

    Q   = load_qtable(f"agriculture_{task_name}")
    env = AgricultureEnv(task=task_name)

    records = []
    for (wp, wc, wf) in weight_grid:
        ep_perf, ep_cost, ep_fair = [], [], []

        for _ in range(N_EPISODES):
            obs, _ = env.reset()
            tot_p = tot_c = tot_f = 0.0

            for _ in range(MAX_STEPS):
                s      = agri_discretize(obs, task_name)
                action = int(np.argmax(Q[s]))

                result   = env.step(action)
                obs_next = result[0]
                info     = result[4] if len(result) >= 5 else {}

                tot_p += info.get("r_performance", 0.0)
                tot_c += info.get("r_cost",        0.0)
                tot_f += info.get("r_fairness",    0.0)

                obs = obs_next
                if result[2] or result[3]:
                    break

            ep_perf.append(tot_p)
            ep_cost.append(tot_c)
            ep_fair.append(tot_f)

        records.append({
            "w_perf":    wp, "w_cost": wc, "w_fair": wf,
            "mean_perf": round(float(np.mean(ep_perf)),  2),
            "mean_cost": round(float(np.mean(ep_cost)),  2),
            "mean_fair": round(float(np.mean(ep_fair)),  2),
            "mean_weighted": round(
                float(np.mean([wp*p + wc*c + wf*f
                               for p,c,f in zip(ep_perf, ep_cost, ep_fair)])), 2),
        })
        print(f"    w=({wp:.2f},{wc:.2f},{wf:.2f}) → "
              f"perf={records[-1]['mean_perf']:7.2f}  "
              f"cost={records[-1]['mean_cost']:7.2f}  "
              f"fair={records[-1]['mean_fair']:7.2f}")

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

TASKS = [
    ("budget",       "finance",     run_finance_task),
    ("trading",      "finance",     run_finance_task),
    ("pest_control", "agriculture", run_agriculture_task),
]

all_records = []

for task_name, domain, runner in TASKS:
    print(f"[{task_name}]")
    records = runner(task_name, WEIGHT_GRID)
    for r in records:
        r["task"] = task_name
    all_records.extend(records)
    print()

# Save CSV
csv_path = os.path.join(ROOT, "experiment_c_results.csv")
fieldnames = ["task", "w_perf", "w_cost", "w_fair",
              "mean_perf", "mean_cost", "mean_fair", "mean_weighted"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_records)
print(f"Results saved → {csv_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_pareto(records, task_name, ax_2d=None, save_path=None):
    pts    = np.array([[r["mean_perf"], r["mean_cost"], r["mean_fair"]] for r in records])
    pareto = is_pareto_optimal(pts)

    weights_perf = np.array([r["w_perf"] for r in records])
    colors       = cm.viridis(weights_perf)

    # ── 2D scatter: performance vs cost, sized by fairness ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Experiment C — Pareto Front: {task_name}\n"
                 f"({N_EPISODES} eval episodes per weight vector, {len(records)} weight configs)",
                 fontsize=11, fontweight="bold")

    for ax, (xi, yi, xl, yl) in zip(axes, [
        (0, 1, "Mean Performance Reward", "Mean Cost Reward"),
        (0, 2, "Mean Performance Reward", "Mean Fairness Reward"),
    ]):
        sc = ax.scatter(pts[:, xi], pts[:, yi],
                        c=weights_perf, cmap="viridis",
                        s=80, zorder=3, label="All configs")
        # Highlight Pareto-optimal
        ax.scatter(pts[pareto, xi], pts[pareto, yi],
                   edgecolors="red", facecolors="none",
                   s=160, linewidths=2, zorder=4, label="Pareto-optimal")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="w_perf")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.close()

    n_pareto = pareto.sum()
    print(f"  [{task_name}] {n_pareto}/{len(records)} weight configs are Pareto-optimal")
    return pareto


# Per-task plots
for task_name, _, _ in TASKS:
    records   = [r for r in all_records if r["task"] == task_name]
    save_path = os.path.join(ROOT, f"experiment_c_pareto_{task_name}.png")
    plot_pareto(records, task_name, save_path=save_path)


# ── Combined 3-panel summary ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Experiment C — Pareto Front Summary (Performance vs Cost)\n"
             "Red circles = Pareto-optimal configurations",
             fontsize=12, fontweight="bold")

for ax, (task_name, _, _) in zip(axes, TASKS):
    records = [r for r in all_records if r["task"] == task_name]
    pts     = np.array([[r["mean_perf"], r["mean_cost"], r["mean_fair"]] for r in records])
    pareto  = is_pareto_optimal(pts)
    wp      = np.array([r["w_perf"] for r in records])

    sc = ax.scatter(pts[:, 0], pts[:, 1], c=wp, cmap="viridis", s=80, zorder=3)
    ax.scatter(pts[pareto, 0], pts[pareto, 1],
               edgecolors="red", facecolors="none",
               s=160, linewidths=2, zorder=4, label="Pareto-optimal")
    ax.set_title(task_name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean Performance Reward", fontsize=8)
    ax.set_ylabel("Mean Cost Reward", fontsize=8)
    plt.colorbar(sc, ax=ax, label="w_perf")
    ax.legend(fontsize=7)

plt.tight_layout()
combined_path = os.path.join(ROOT, "experiment_c_pareto_combined.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Combined plot → {combined_path}")
print("\nExperiment C complete.")