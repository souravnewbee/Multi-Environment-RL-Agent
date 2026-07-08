# =============================================================================
# UMORDA — Energy Domain Graphical Training Visualizer
# File: visualize_energy.py
# =============================================================================

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.dirname(__file__))
from environments.energy_env import EnergyEnv
from agents.energy_agent import EnergyAgent

EPISODES      = 20_000
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.9995
TASKS         = ["solar_scheduling", "battery_management", "grid_interaction"]
QTABLE_DIR    = os.path.join(os.path.dirname(__file__), "qtables")

BG_COLOR    = "#0d1117"
PANEL_COLOR = "#161b22"
GRID_COLOR  = "#21262d"
TEXT_COLOR  = "#e6edf3"
ACCENT      = "#58a6ff"
GREEN       = "#3fb950"
ORANGE      = "#d29922"
RED         = "#f85149"
PURPLE      = "#bc8cff"
YELLOW      = "#e3b341"
TASK_COLORS = [ACCENT, GREEN, ORANGE]


def train_and_collect(task):
    env   = EnergyEnv(task=task)
    agent = EnergyAgent(env.observation_space.n, env.action_space.n,
                        alpha=ALPHA, gamma=GAMMA,
                        epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
                        epsilon_decay=EPSILON_DECAY)
    rewards, epsilons, avgs = [], [], []
    print(f"  Training [{task}] ... ", end="", flush=True)

    for ep in range(1, EPISODES+1):
        state,_ = env.reset()
        total,done,steps = 0.0,False,0
        while not done and steps < 50:
            a = agent.select_action(state)
            ns,r,t,tr,_ = env.step(a)
            agent.update(state,a,r,ns)
            state,total,done,steps = ns,total+r,t or tr,steps+1
        agent.decay_epsilon()
        rewards.append(total)
        epsilons.append(agent.epsilon)
        window = 500
        avgs.append(np.mean(rewards[-window:]) if ep >= window else np.mean(rewards))

    print(f"Done! Avg={np.mean(rewards):.2f}")
    os.makedirs(QTABLE_DIR, exist_ok=True)
    agent.save_qtable(os.path.join(QTABLE_DIR, f"energy_{task}_qtable.npy"))
    return agent, rewards, epsilons, avgs


def main():
    print("\n" + "="*60)
    print("  UMORDA — ENERGY DOMAIN GRAPHICAL TRAINING")
    print("  Balcony Solar Panel Optimization")
    print("="*60)

    all_results = {}
    for task in TASKS:
        agent, rewards, epsilons, avgs = train_and_collect(task)
        all_results[task] = {
            "agent": agent, "rewards": rewards,
            "epsilons": epsilons, "avgs": avgs,
            "cfg": EnergyEnv.TASK_CONFIG[task],
        }

    episodes = np.arange(1, EPISODES+1)

    # ── Figure 1: Reward curves + Epsilon decay ──────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.patch.set_facecolor(BG_COLOR)
    fig1.suptitle("UMORDA — Energy Domain Q-Learning Training (20,000 Episodes)\n"
                  "Balcony Solar Panel (Balkonkraftwerk) Optimization",
                  fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)

    for col, (task, color) in enumerate(zip(TASKS, TASK_COLORS)):
        res     = all_results[task]
        rewards = res["rewards"]
        avgs    = res["avgs"]
        eps     = res["epsilons"]
        cfg     = res["cfg"]

        ax = axes[0][col]
        ax.set_facecolor(PANEL_COLOR)
        ax.plot(episodes, rewards, color=color, alpha=0.2, linewidth=0.5, label="Episode Reward")
        ax.plot(episodes, avgs,    color=color, alpha=1.0, linewidth=2.0, label="Avg (500 ep)")
        ax.set_title(f"{task.upper()}\n{cfg['description']}", color=TEXT_COLOR,
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Total Reward", color=TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5)
        ax.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID_COLOR)
        ax.annotate(f"Final Avg: {avgs[-1]:.1f}", xy=(EPISODES, avgs[-1]),
                    xytext=(-100, 20), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color))

        ax2 = axes[1][col]
        ax2.set_facecolor(PANEL_COLOR)
        ax2.plot(episodes, eps, color=PURPLE, linewidth=2.0)
        ax2.fill_between(episodes, eps, alpha=0.15, color=PURPLE)
        ax2.axhline(y=EPSILON_MIN, color=RED, linestyle="--", linewidth=1.2,
                    label=f"Min ε={EPSILON_MIN}")
        ax2.set_title(f"Epsilon Decay — {task.upper()}", color=TEXT_COLOR,
                      fontsize=10, fontweight="bold")
        ax2.set_xlabel("Episode", color=TEXT_COLOR, fontsize=9)
        ax2.set_ylabel("Epsilon (ε)", color=TEXT_COLOR, fontsize=9)
        ax2.tick_params(colors=TEXT_COLOR)
        ax2.grid(color=GRID_COLOR, linewidth=0.5)
        ax2.legend(fontsize=8, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax2.spines.values(): spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(QTABLE_DIR, "energy_training_overview.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("\n  [✓] Figure 1 saved: energy_training_overview.png")

    # ── Figure 2: Q-Table Heatmaps ────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.patch.set_facecolor(BG_COLOR)
    fig2.suptitle("UMORDA — Trained Q-Tables Heatmap (Energy Domain)\n"
                  "Balcony Solar Panel Optimization",
                  fontsize=14, fontweight="bold", color=TEXT_COLOR)

    cmap = LinearSegmentedColormap.from_list("energy",
           ["#0d1117", "#1f6feb", "#58a6ff", "#3fb950", "#e3b341"])

    for col, task in enumerate(TASKS):
        agent  = all_results[task]["agent"]
        cfg    = all_results[task]["cfg"]
        qtable = agent.q_table
        display_rows = min(40, qtable.shape[0])

        ax = axes2[col]
        ax.set_facecolor(PANEL_COLOR)
        im = ax.imshow(qtable[:display_rows], aspect="auto", cmap=cmap)
        ax.set_title(f"Q-Table: {task.upper()}\n({qtable.shape[0]} states × {qtable.shape[1]} actions)",
                     color=TEXT_COLOR, fontsize=10, fontweight="bold")
        ax.set_xlabel("Actions", color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel("State Index", color=TEXT_COLOR, fontsize=9)
        ax.set_xticks(range(len(cfg["action_meanings"])))
        ax.set_xticklabels([a[:12] for a in cfg["action_meanings"]],
                           color=TEXT_COLOR, fontsize=7, rotation=15)
        ax.tick_params(colors=TEXT_COLOR)
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.set_label("Q-Value", color=TEXT_COLOR, fontsize=9)
        best_actions = np.argmax(qtable[:display_rows], axis=1)
        for row in range(display_rows):
            ax.add_patch(plt.Rectangle(
                (best_actions[row]-0.5, row-0.5), 1, 1,
                fill=False, edgecolor=ORANGE, linewidth=1.5))

    plt.tight_layout()
    fig2.savefig(os.path.join(QTABLE_DIR, "energy_qtable_heatmap.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("  [✓] Figure 2 saved: energy_qtable_heatmap.png")

    # ── Figure 3: Summary Dashboard ───────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    fig3.patch.set_facecolor(BG_COLOR)
    ax3.set_facecolor(PANEL_COLOR)

    x     = np.arange(len(TASKS))
    width = 0.35
    avgs  = [np.mean(all_results[t]["rewards"]) for t in TASKS]
    bests = [max(all_results[t]["rewards"]) for t in TASKS]

    bars1 = ax3.bar(x - width/2, avgs,  width, label="Avg Reward",
                    color=TASK_COLORS, alpha=0.85, edgecolor=TEXT_COLOR, linewidth=0.5)
    bars2 = ax3.bar(x + width/2, bests, width, label="Best Episode",
                    color=TASK_COLORS, alpha=0.45, edgecolor=TEXT_COLOR, linewidth=0.5)

    for bar in bars1:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 color=TEXT_COLOR, fontsize=9, fontweight="bold")
    for bar in bars2:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 color=TEXT_COLOR, fontsize=9)

    ax3.set_title("Training Summary — All Energy Tasks (20,000 Episodes)\n"
                  "Balcony Solar Panel Optimization",
                  color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    labels = ["Solar\nScheduling", "Battery\nManagement", "Grid\nInteraction"]
    ax3.set_xticklabels(labels, color=TEXT_COLOR, fontsize=10)
    ax3.set_ylabel("Reward", color=TEXT_COLOR, fontsize=10)
    ax3.tick_params(colors=TEXT_COLOR)
    ax3.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax3.legend(fontsize=10, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax3.spines.values(): spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    fig3.savefig(os.path.join(QTABLE_DIR, "energy_summary_dashboard.png"),
                 dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print("  [✓] Figure 3 saved: energy_summary_dashboard.png")

    plt.show()
    print("\n  All graphs displayed! Close windows to exit.\n")


if __name__ == "__main__":
    main()
