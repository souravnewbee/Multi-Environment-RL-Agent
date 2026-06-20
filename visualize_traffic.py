# =============================================================================
# UMORDA — Traffic Domain Graphical Training Visualizer
# File: visualize_traffic.py
# Usage: python visualize_traffic.py
# Shows: Reward curves, Epsilon decay, Q-Table heatmaps for all 3 tasks
# =============================================================================

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.dirname(__file__))

from environments.traffic_env import TrafficEnv
from agents.traffic_agent import TrafficAgent

# =============================================================================
# CONFIG
# =============================================================================
EPISODES      = 20_000
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.9995
TASKS         = ["intersection", "pedestrian", "parking"]
QTABLE_DIR    = os.path.join(os.path.dirname(__file__), "qtables")

# Custom color theme
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


# =============================================================================
# TRAINING FUNCTION (with history collection)
# =============================================================================
def train_and_collect(task: str):
    env   = TrafficEnv(task=task)
    cfg   = TrafficEnv.TASK_CONFIG[task]
    agent = TrafficAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    reward_history  = []
    epsilon_history = []
    avg_history     = []

    print(f"  Training [{task}] ... ", end="", flush=True)

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            done = terminated or truncated
        agent.decay_epsilon()
        reward_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        window = 500
        if episode >= window:
            avg_history.append(np.mean(reward_history[-window:]))
        else:
            avg_history.append(np.mean(reward_history))

    print(f"Done! Avg reward: {np.mean(reward_history):.2f}")

    os.makedirs(QTABLE_DIR, exist_ok=True)
    agent.save_qtable(os.path.join(QTABLE_DIR, f"traffic_{task}_qtable.npy"))

    return agent, reward_history, epsilon_history, avg_history


# =============================================================================
# SMOOTHING
# =============================================================================
def smooth(data, window=200):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================
def main():
    print("\n" + "="*60)
    print("  UMORDA — TRAFFIC DOMAIN GRAPHICAL TRAINING")
    print("="*60)

    # --- Train all tasks ---
    all_results = {}
    for task in TASKS:
        agent, rewards, epsilons, avgs = train_and_collect(task)
        all_results[task] = {
            "agent":    agent,
            "rewards":  rewards,
            "epsilons": epsilons,
            "avgs":     avgs,
            "cfg":      TrafficEnv.TASK_CONFIG[task],
        }

    episodes = np.arange(1, EPISODES + 1)

    # =========================================================================
    # FIGURE 1 — Training Overview (Rewards + Epsilon)
    # =========================================================================
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.patch.set_facecolor(BG_COLOR)
    fig1.suptitle(
        "UMORDA — Traffic Domain Q-Learning Training  (20,000 Episodes)",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98
    )

    for col, (task, color) in enumerate(zip(TASKS, TASK_COLORS)):
        res = all_results[task]
        rewards  = res["rewards"]
        avgs     = res["avgs"]
        epsilons = res["epsilons"]
        cfg      = res["cfg"]

        # --- Row 0: Reward curve ---
        ax = axes[0][col]
        ax.set_facecolor(PANEL_COLOR)
        ax.plot(episodes, rewards,  color=color,  alpha=0.25, linewidth=0.5, label="Episode Reward")
        ax.plot(episodes, avgs,     color=color,  alpha=1.0,  linewidth=2.0, label="Avg (500 ep)")
        ax.set_title(f"{task.upper()}\n{cfg['description']}", color=TEXT_COLOR, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Total Reward", color=TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=TEXT_COLOR)
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5)
        ax.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        # Annotate final avg
        final_avg = avgs[-1]
        ax.annotate(
            f"Final Avg: {final_avg:.1f}",
            xy=(EPISODES, final_avg),
            xytext=(-120, 20), textcoords="offset points",
            color=color, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
        )

        # --- Row 1: Epsilon decay ---
        ax2 = axes[1][col]
        ax2.set_facecolor(PANEL_COLOR)
        ax2.plot(episodes, epsilons, color=PURPLE, linewidth=2.0)
        ax2.fill_between(episodes, epsilons, alpha=0.15, color=PURPLE)
        ax2.axhline(y=EPSILON_MIN, color=RED, linestyle="--", linewidth=1.2, label=f"Min ε = {EPSILON_MIN}")
        ax2.set_title(f"Epsilon Decay — {task.upper()}", color=TEXT_COLOR, fontsize=11, fontweight="bold")
        ax2.set_xlabel("Episode", color=TEXT_COLOR, fontsize=9)
        ax2.set_ylabel("Epsilon (ε)", color=TEXT_COLOR, fontsize=9)
        ax2.tick_params(colors=TEXT_COLOR)
        ax2.spines[:].set_color(GRID_COLOR)
        ax2.grid(color=GRID_COLOR, linewidth=0.5)
        ax2.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax2.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        # Mark convergence point
        conv_ep = next((i for i, e in enumerate(epsilons) if e <= EPSILON_MIN + 0.001), EPISODES-1)
        ax2.axvline(x=conv_ep, color=ORANGE, linestyle=":", linewidth=1.5)
        ax2.annotate(
            f"Converged\nEp {conv_ep:,}",
            xy=(conv_ep, epsilons[conv_ep]),
            xytext=(30, 20), textcoords="offset points",
            color=ORANGE, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2)
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(os.path.join(QTABLE_DIR, "traffic_training_overview.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("\n  [✓] Figure 1 saved: traffic_training_overview.png")

    # =========================================================================
    # FIGURE 2 — Q-Table Heatmaps
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.patch.set_facecolor(BG_COLOR)
    fig2.suptitle(
        "UMORDA — Trained Q-Tables Heatmap (Traffic Domain)",
        fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.01
    )

    cmap = LinearSegmentedColormap.from_list("umorda", ["#0d1117", "#1f6feb", "#58a6ff", "#3fb950"])

    for col, task in enumerate(TASKS):
        res   = all_results[task]
        agent = res["agent"]
        cfg   = res["cfg"]
        qtable = agent.q_table

        ax = axes2[col]
        ax.set_facecolor(PANEL_COLOR)

        # Show first 30 states max for readability
        display_rows = min(30, qtable.shape[0])
        im = ax.imshow(qtable[:display_rows], aspect="auto", cmap=cmap)

        ax.set_title(f"Q-Table: {task.upper()}\n({qtable.shape[0]} states × {qtable.shape[1]} actions)",
                     color=TEXT_COLOR, fontsize=11, fontweight="bold")
        ax.set_xlabel("Actions", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("State Index", color=TEXT_COLOR, fontsize=9)
        ax.set_xticks(range(len(cfg["action_meanings"])))
        ax.set_xticklabels(
            [a.replace(" ", "\n") for a in cfg["action_meanings"]],
            color=TEXT_COLOR, fontsize=8
        )
        ax.tick_params(colors=TEXT_COLOR)

        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.set_label("Q-Value", color=TEXT_COLOR, fontsize=9)

        # Annotate best action per state
        best_actions = np.argmax(qtable[:display_rows], axis=1)
        for row in range(display_rows):
            ax.add_patch(plt.Rectangle(
                (best_actions[row] - 0.5, row - 0.5), 1, 1,
                fill=False, edgecolor=ORANGE, linewidth=1.5
            ))

    plt.tight_layout()
    fig2.savefig(os.path.join(QTABLE_DIR, "traffic_qtable_heatmap.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("  [✓] Figure 2 saved: traffic_qtable_heatmap.png")

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

    bars1 = ax3.bar(x - width/2, avg_rewards, width, label="Avg Reward",
                    color=[ACCENT, GREEN, ORANGE], alpha=0.85, edgecolor=TEXT_COLOR, linewidth=0.5)
    bars2 = ax3.bar(x + width/2, max_rewards, width, label="Best Episode",
                    color=[ACCENT, GREEN, ORANGE], alpha=0.45, edgecolor=TEXT_COLOR, linewidth=0.5)

    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 color=TEXT_COLOR, fontsize=9, fontweight="bold")
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 color=TEXT_COLOR, fontsize=9)

    ax3.set_title("Training Summary — All Traffic Tasks (20,000 Episodes)",
                  color=TEXT_COLOR, fontsize=13, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([t.upper() for t in TASKS], color=TEXT_COLOR, fontsize=11)
    ax3.set_ylabel("Reward", color=TEXT_COLOR, fontsize=10)
    ax3.tick_params(colors=TEXT_COLOR)
    ax3.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax3.legend(fontsize=10, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    fig3.savefig(os.path.join(QTABLE_DIR, "traffic_summary_dashboard.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("  [✓] Figure 3 saved: traffic_summary_dashboard.png")

    # --- Show all figures ---
    plt.show()
    print("\n  All graphs displayed! Close graph windows to exit.\n")


if __name__ == "__main__":
    main()
