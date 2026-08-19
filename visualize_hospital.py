# =============================================================================
# UMORDA — Hospital Domain Graphical Training Visualizer
# File: visualize_hospital.py
# Usage: python visualize_hospital.py
# Shows: Reward curves, Epsilon decay, and a learned-policy heatmap for
#        each of the 3 hospital tasks (bed_allocation, er_queue, staff_allocation)
#
# NOTE on shape: unlike the traffic/energy agents (flat n_states x n_actions
# Q-tables indexed by a single encoded state integer), HospitalAgent keeps a
# multi-dimensional Q-table per task, e.g. bed_allocation is
# (free_beds_bins, waiting_patients_bins, n_actions) = (6, 7, 3). That's
# actually a gift for visualization: instead of a raw state-index heatmap,
# Figure 2 below plots the *learned policy* directly on its two real state
# axes (e.g. free beds vs. waiting patients), with each cell colored by the
# chosen action — no reshaping/guessing required.
# =============================================================================

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(__file__))
from environments.hospital_env import HospitalEnv
from agents.hospital_agent import HospitalAgent, Q_SHAPES

# =============================================================================
# CONFIG — matches training/train_hospital.py exactly (same episode count,
# same per-task alpha/gamma/epsilon_decay) so this produces the same policy
# your official training run does, not an approximation.
# =============================================================================
EPISODES      = 20_000     # each "episode" is a full 60-step shift
EPSILON_START = 1.0
EPSILON_MIN   = 0.01

TASKS = ["bed_allocation", "er_queue", "staff_allocation"]

TASK_HYPERPARAMS = {
    "bed_allocation":   {"n_actions": 3, "alpha": 0.1,  "gamma": 0.95, "epsilon_decay": 0.9995},
    "er_queue":         {"n_actions": 2, "alpha": 0.1,  "gamma": 0.95, "epsilon_decay": 0.9995},
    "staff_allocation": {"n_actions": 3, "alpha": 0.08, "gamma": 0.95, "epsilon_decay": 0.9997},
}

# Per-task info needed to label a 2D policy heatmap: which raw obs indices
# map to which axis, how those are binned by discretize(), and axis titles.
TASK_AXES = {
    "bed_allocation": {
        "row_label": "Free beds",       "row_bin_size": 4,  "row_cap": 20,
        "col_label": "Waiting patients", "col_bin_size": 10, "col_cap": 60,
        "actions": ["Admit", "Reject", "Transfer"],
    },
    "er_queue": {
        "row_label": "Emergency queue", "row_bin_size": 4,  "row_cap": 20,
        "col_label": "Normal queue",    "col_bin_size": 8,  "col_cap": 40,
        "actions": ["Serve Emergency", "Serve Normal"],
    },
    "staff_allocation": {
        "row_label": "Available doctors", "row_bin_size": 4,  "row_cap": 20,
        "col_label": "Patient load",      "col_bin_size": 16, "col_cap": 80,
        "actions": ["Assign More", "Keep Current", "Reduce Staff"],
    },
}

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "qtables")

BG_COLOR    = "#0d1117"
PANEL_COLOR = "#161b22"
GRID_COLOR  = "#21262d"
TEXT_COLOR  = "#e6edf3"
ACCENT      = "#58a6ff"
GREEN       = "#3fb950"
ORANGE      = "#d29922"
RED         = "#f85149"
PURPLE      = "#bc8cff"
TASK_COLORS = [ACCENT, GREEN, ORANGE]

# Up to 3 distinct action colors per task (reused per-task, not global)
ACTION_PALETTE = ["#3fb950", "#f85149", "#58a6ff", "#e3b341"]


def _action_labels(actions):
    """Short, mutually-distinct cell labels. Falls back from word-initials
    (e.g. 'Serve Emergency' -> 'SE') to a longer prefix if that still
    collides (e.g. two actions sharing initials)."""
    labels, seen = [], set()
    for a in actions:
        words = a.split()
        label = "".join(w[0] for w in words).upper() if len(words) > 1 else a[:3].upper()
        if label in seen:
            label = a[:4].upper()
        seen.add(label)
        labels.append(label)
    return labels


# =============================================================================
# TRAINING FUNCTION (with history collection)
# =============================================================================
def train_and_collect(task: str):
    env = HospitalEnv(task=task)
    q_shape = Q_SHAPES[task]
    hp = TASK_HYPERPARAMS[task]

    agent = HospitalAgent(
        task, q_shape, hp["n_actions"],
        alpha=hp["alpha"], gamma=hp["gamma"],
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=hp["epsilon_decay"],
    )

    reward_history = []
    epsilon_history = []
    avg_history = []

    print(f"  Training [{task}] ... ", end="", flush=True)

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        deltas = []

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            delta = agent.update(obs, action, reward, next_obs, terminated or truncated)
            deltas.append(delta)
            obs = next_obs
            total_reward += reward
            done = terminated or truncated

        agent.decay_epsilon()
        agent.log_episode(total_reward, float(np.mean(deltas)) if deltas else 0.0)
        reward_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        window = 500
        if episode >= window:
            avg_history.append(np.mean(reward_history[-window:]))
        else:
            avg_history.append(np.mean(reward_history))

    print(f"Done! Avg reward: {np.mean(reward_history):.2f}")

    os.makedirs(QTABLE_DIR, exist_ok=True)
    try:
        agent.save()
    except Exception as e:
        print(f"  [!] Could not save Q-table for {task} (qtable_store unavailable): {e}")

    return agent, reward_history, epsilon_history, avg_history


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("  UMORDA — HOSPITAL DOMAIN GRAPHICAL TRAINING")
    print("=" * 60)

    all_results = {}
    for task in TASKS:
        agent, rewards, epsilons, avgs = train_and_collect(task)
        all_results[task] = {
            "agent": agent,
            "rewards": rewards,
            "epsilons": epsilons,
            "avgs": avgs,
        }

    episodes = np.arange(1, EPISODES + 1)

    # =========================================================================
    # FIGURE 1 — Training Overview (Rewards + Epsilon)
    # =========================================================================
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.patch.set_facecolor(BG_COLOR)
    fig1.suptitle(
        f"UMORDA — Hospital Domain Q-Learning Training ({EPISODES:,} Shifts)",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98
    )

    for col, (task, color) in enumerate(zip(TASKS, TASK_COLORS)):
        res = all_results[task]
        rewards, avgs, epsilons = res["rewards"], res["avgs"], res["epsilons"]

        ax = axes[0][col]
        ax.set_facecolor(PANEL_COLOR)
        ax.plot(episodes, rewards, color=color, alpha=0.25, linewidth=0.5, label="Shift Reward")
        ax.plot(episodes, avgs, color=color, alpha=1.0, linewidth=2.0, label="Avg (500 shifts)")
        ax.set_title(task.upper().replace("_", " "), color=TEXT_COLOR, fontsize=11, fontweight="bold")
        ax.set_xlabel("Shift (episode)", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Total Reward", color=TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5)
        ax.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        final_avg = avgs[-1]
        ax.annotate(
            f"Final Avg: {final_avg:.1f}",
            xy=(EPISODES, final_avg),
            xytext=(-120, 20), textcoords="offset points",
            color=color, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
        )

        ax2 = axes[1][col]
        ax2.set_facecolor(PANEL_COLOR)
        ax2.plot(episodes, epsilons, color=PURPLE, linewidth=2.0)
        ax2.fill_between(episodes, epsilons, alpha=0.15, color=PURPLE)
        ax2.axhline(y=EPSILON_MIN, color=RED, linestyle="--", linewidth=1.2, label=f"Min ε = {EPSILON_MIN}")
        ax2.set_title(f"Epsilon Decay — {task.upper().replace('_', ' ')}", color=TEXT_COLOR,
                      fontsize=11, fontweight="bold")
        ax2.set_xlabel("Shift (episode)", color=TEXT_COLOR, fontsize=9)
        ax2.set_ylabel("Epsilon (ε)", color=TEXT_COLOR, fontsize=9)
        ax2.tick_params(colors=TEXT_COLOR)
        ax2.grid(color=GRID_COLOR, linewidth=0.5)
        ax2.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(os.path.join(QTABLE_DIR, "hospital_training_overview.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("\n  [✓] Figure 1 saved: hospital_training_overview.png")

    # =========================================================================
    # FIGURE 2 — Learned Policy Heatmaps (best action per 2D state)
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6.5))
    fig2.patch.set_facecolor(BG_COLOR)
    fig2.suptitle(
        "UMORDA — Learned Policy (Hospital Domain)\nbest action per state, highlighted per cell",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.03
    )

    for col, task in enumerate(TASKS):
        agent = all_results[task]["agent"]
        Q = agent.Q  # shape (rows, cols, n_actions)
        cfg = TASK_AXES[task]
        actions = cfg["actions"]
        n_rows, n_cols = Q.shape[0], Q.shape[1]

        best_action = np.argmax(Q, axis=2)
        colors = ACTION_PALETTE[:len(actions)]
        cmap = ListedColormap(colors)
        action_labels = _action_labels(actions)

        ax = axes2[col]
        ax.set_facecolor(PANEL_COLOR)
        ax.imshow(best_action, cmap=cmap, vmin=-0.5, vmax=len(actions) - 0.5, aspect="auto")

        for r in range(n_rows):
            for c in range(n_cols):
                a = best_action[r, c]
                ax.text(c, r, action_labels[a], ha="center", va="center",
                        color="#0d1117", fontsize=9, fontweight="bold")
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                            edgecolor=TEXT_COLOR, linewidth=0.5, alpha=0.35))

        row_labels = [f"{r * cfg['row_bin_size']}-{r * cfg['row_bin_size'] + cfg['row_bin_size'] - 1}"
                      for r in range(n_rows)]
        row_labels[-1] = f"{cfg['row_cap']}+"
        col_labels = [f"{c * cfg['col_bin_size']}-{c * cfg['col_bin_size'] + cfg['col_bin_size'] - 1}"
                      for c in range(n_cols)]
        col_labels[-1] = f"{cfg['col_cap']}+"

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, color=TEXT_COLOR, fontsize=8, rotation=30)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, color=TEXT_COLOR, fontsize=8)
        ax.set_xlabel(cfg["col_label"], color=TEXT_COLOR, fontsize=10, fontweight="bold")
        ax.set_ylabel(cfg["row_label"], color=TEXT_COLOR, fontsize=10, fontweight="bold")
        ax.set_title(task.upper().replace("_", " "), color=TEXT_COLOR, fontsize=11, fontweight="bold")
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        legend_elems = [Patch(facecolor=colors[i], label=actions[i]) for i in range(len(actions))]
        leg = ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.32),
                         ncol=len(actions), facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, fontsize=8)
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)

    plt.tight_layout()
    fig2.savefig(os.path.join(QTABLE_DIR, "hospital_policy_heatmap.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("  [✓] Figure 2 saved: hospital_policy_heatmap.png")

    # =========================================================================
    # FIGURE 3 — Final Summary Dashboard
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    fig3.patch.set_facecolor(BG_COLOR)
    ax3.set_facecolor(PANEL_COLOR)

    x = np.arange(len(TASKS))
    width = 0.35
    avg_rewards = [np.mean(all_results[t]["rewards"]) for t in TASKS]
    max_rewards = [max(all_results[t]["rewards"]) for t in TASKS]

    bars1 = ax3.bar(x - width / 2, avg_rewards, width, label="Avg Reward",
                     color=TASK_COLORS, alpha=0.85, edgecolor=TEXT_COLOR, linewidth=0.5)
    bars2 = ax3.bar(x + width / 2, max_rewards, width, label="Best Shift",
                     color=TASK_COLORS, alpha=0.45, edgecolor=TEXT_COLOR, linewidth=0.5)

    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{bar.get_height():.1f}", ha="center", va="bottom",
                  color=TEXT_COLOR, fontsize=9, fontweight="bold")
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{bar.get_height():.1f}", ha="center", va="bottom",
                  color=TEXT_COLOR, fontsize=9)

    ax3.set_title(f"Training Summary — All Hospital Tasks ({EPISODES:,} Shifts)",
                  color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([t.upper().replace("_", "\n") for t in TASKS], color=TEXT_COLOR, fontsize=10)
    ax3.set_ylabel("Reward", color=TEXT_COLOR, fontsize=10)
    ax3.tick_params(colors=TEXT_COLOR)
    ax3.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax3.legend(fontsize=10, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    fig3.savefig(os.path.join(QTABLE_DIR, "hospital_summary_dashboard.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("  [✓] Figure 3 saved: hospital_summary_dashboard.png")

    plt.show()
    print("\n  All graphs displayed! Close graph windows to exit.\n")


if __name__ == "__main__":
    main()
