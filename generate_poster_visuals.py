"""
UMORDA — Poster Visual Generator
File: generate_poster_visuals.py

Generates TWO poster-ready PNGs:

  1. poster_reward_curve.png
     Single-panel Q-learning training curve (raw + smoothed reward,
     epsilon decay on a secondary axis) for ONE chosen task.

  2. poster_trained_vs_random.png
     Grouped bar chart comparing trained-agent reward vs random-baseline
     reward across all 15 tasks (or domain-averaged if TASKS_PER_DOMAIN
     view is too cramped — see USE_DOMAIN_AVERAGE flag below).

Run from the project root (same folder as qtable_store.py):

    python generate_poster_visuals.py

Requirements: numpy, matplotlib, gymnasium (same as the rest of the repo).
Q-tables are read from qtables/qtables.db via qtable_store.py — train
whatever's missing first (training/train_*.py) or this will skip that
task and print a warning instead of crashing.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qtable_store import load_qtable

# =============================================================================
# CONFIG — tweak these two lines to change what gets plotted
# =============================================================================
CURVE_DOMAIN = "traffic"       # domain to train live for the reward-curve panel
CURVE_TASK   = "pedestrian"    # task within that domain (clean convergence story)
CURVE_EPISODES = 20_000

USE_DOMAIN_AVERAGE = False     # True -> 5 bars (one per domain), False -> up to 15 bars
EVAL_EPISODES = 200            # episodes per task when evaluating trained vs random

OUT_CURVE = "poster_reward_curve.png"
OUT_BAR   = "poster_trained_vs_random.png"


# =============================================================================
# PART 1 — Reward curve + epsilon overlay for ONE task
# =============================================================================
def generate_reward_curve():
    print(f"\n[1/2] Training live for reward-curve panel: {CURVE_DOMAIN}/{CURVE_TASK} "
          f"({CURVE_EPISODES:,} episodes)...")

    if CURVE_DOMAIN == "traffic":
        from environments.traffic_env import TrafficEnv
        from agents.traffic_agent import TrafficAgent
        env   = TrafficEnv(task=CURVE_TASK)
        agent = TrafficAgent(env.observation_space.n, env.action_space.n,
                              alpha=0.1, gamma=0.95,
                              epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995)
    elif CURVE_DOMAIN == "energy":
        from environments.energy_env import EnergyEnv
        from agents.energy_agent import EnergyAgent
        env   = EnergyEnv(task=CURVE_TASK)
        agent = EnergyAgent(env.observation_space.n, env.action_space.n,
                             alpha=0.1, gamma=0.95,
                             epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995)
    else:
        raise ValueError(f"CURVE_DOMAIN must be 'traffic' or 'energy' for this script, got '{CURVE_DOMAIN}'")

    rewards, epsilons, avgs = [], [], []
    window = 500

    for ep in range(1, CURVE_EPISODES + 1):
        state, _ = env.reset()
        total, done = 0.0, False
        while not done:
            a = agent.select_action(state)
            ns, r, term, trunc, _ = env.step(a)
            agent.update(state, a, r, ns)
            state, total, done = ns, total + r, (term or trunc)
        agent.decay_epsilon()
        rewards.append(total)
        epsilons.append(agent.epsilon)
        avgs.append(np.mean(rewards[-window:]) if ep >= window else np.mean(rewards))

        if ep % 2000 == 0:
            print(f"    Episode {ep:>6,} | avg reward: {avgs[-1]:.2f} | epsilon: {agent.epsilon:.4f}")

    episodes = np.arange(1, CURVE_EPISODES + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(episodes, rewards, color="#4C72B0", alpha=0.15, linewidth=0.5, label="Episode reward")
    ax1.plot(episodes, avgs,    color="#4C72B0", linewidth=2.5, label=f"Avg reward ({window}-ep window)")
    ax1.set_xlabel("Training Episode", fontsize=13)
    ax1.set_ylabel("Reward", fontsize=13, color="#4C72B0")
    ax1.tick_params(axis='y', labelcolor="#4C72B0")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilons, color="#C44E52", linestyle="--", linewidth=2, label="Epsilon (ε)")
    ax2.set_ylabel("Epsilon (exploration rate)", fontsize=13, color="#C44E52")
    ax2.tick_params(axis='y', labelcolor="#C44E52")

    ax1.set_title(f"Q-Learning Convergence — {CURVE_DOMAIN.title()} / {CURVE_TASK.replace('_',' ').title()}",
                  fontsize=15, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10, frameon=True)

    ax1.annotate(f"Converged avg: {avgs[-1]:.1f}",
                 xy=(CURVE_EPISODES, avgs[-1]), xytext=(-160, -35), textcoords="offset points",
                 fontsize=11, fontweight="bold", color="#4C72B0",
                 arrowprops=dict(arrowstyle="->", color="#4C72B0"))

    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [OK] Saved -> {OUT_CURVE}")


# =============================================================================
# PART 2 — Trained vs Random bar chart across all tasks
# =============================================================================
def _eval_discretized(env, Q, discretize_fn, task, n_episodes, greedy):
    """Gym-style env whose obs needs a discretize() call (hospital, agriculture)."""
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            if greedy:
                s = discretize_fn(obs, task)
                a = int(np.argmax(Q[s]))
            else:
                a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        scores.append(total)
    return float(np.mean(scores))


def _eval_encoded(env, Q, n_episodes, greedy):
    """Gym-style env whose obs IS already the flat Q-table index (traffic, energy)."""
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            if greedy:
                a = int(np.argmax(Q[obs]))
            else:
                a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        scores.append(total)
    return float(np.mean(scores))


def _eval_finance(task, n_episodes, greedy):
    """Custom (non-gym) FinanceEnv: reset() -> dict, step() -> (dict, r, done, info)."""
    from environments.finance_env import FinanceEnv
    from agents.finance_agent import FinanceAgent

    agent = FinanceAgent(task=task)
    scores = []
    for _ in range(n_episodes):
        env = FinanceEnv(task=task, ticker="AAPL") if task == "trading" else FinanceEnv(task=task)
        state = env.reset()
        total, done = 0.0, False
        while not done:
            if greedy:
                action_idx, _, _ = agent.get_action(state)
            else:
                action_idx = np.random.randint(env.n_actions)
            state, r, done, _ = env.step(action_idx)
            total += r
        scores.append(total)
    return float(np.mean(scores))


def generate_trained_vs_random():
    print(f"\n[2/2] Evaluating trained vs random across all tasks "
          f"({EVAL_EPISODES} episodes each)...")

    results = []  # list of (domain, task, trained_mean, random_mean)

    # ── Hospital + Agriculture: gym-style with discretize() helpers ──────────
    try:
        from environments.hospital_env import HospitalEnv
        from agents.hospital_agent import discretize as hosp_discretize
        for task in ["bed_allocation", "er_queue", "staff_allocation"]:
            try:
                Q = load_qtable(f"hospital_{task}")
                env = HospitalEnv(task=task)
                trained = _eval_discretized(env, Q, hosp_discretize, task, EVAL_EPISODES, True)
                random_ = _eval_discretized(env, Q, hosp_discretize, task, EVAL_EPISODES, False)
                results.append(("Hospital", task, trained, random_))
                print(f"    [OK] hospital/{task}: trained={trained:.1f}  random={random_:.1f}")
            except FileNotFoundError:
                print(f"    [SKIP] hospital/{task} — Q-table not trained yet")
    except ImportError as e:
        print(f"    [SKIP] hospital domain — import error: {e}")

    try:
        from environments.agriculture_env import AgricultureEnv
        from agents.agriculture_agent import discretize as agri_discretize
        for task in ["soil_preparation", "irrigation", "pest_control"]:
            try:
                Q = load_qtable(f"agriculture_{task}")
                env = AgricultureEnv(task=task)
                trained = _eval_discretized(env, Q, agri_discretize, task, EVAL_EPISODES, True)
                random_ = _eval_discretized(env, Q, agri_discretize, task, EVAL_EPISODES, False)
                results.append(("Agriculture", task, trained, random_))
                print(f"    [OK] agriculture/{task}: trained={trained:.1f}  random={random_:.1f}")
            except FileNotFoundError:
                print(f"    [SKIP] agriculture/{task} — Q-table not trained yet")
    except ImportError as e:
        print(f"    [SKIP] agriculture domain — import error: {e}")

    # ── Traffic + Energy: gym-style, obs already IS the encoded index ────────
    try:
        from environments.traffic_env import TrafficEnv
        for task in ["intersection", "pedestrian", "parking"]:
            try:
                Q = load_qtable(f"traffic_{task}")
                env = TrafficEnv(task=task)
                trained = _eval_encoded(env, Q, EVAL_EPISODES, True)
                random_ = _eval_encoded(env, Q, EVAL_EPISODES, False)
                results.append(("Traffic", task, trained, random_))
                print(f"    [OK] traffic/{task}: trained={trained:.1f}  random={random_:.1f}")
            except FileNotFoundError:
                print(f"    [SKIP] traffic/{task} — Q-table not trained yet")
    except ImportError as e:
        print(f"    [SKIP] traffic domain — import error: {e}")

    try:
        from environments.energy_env import EnergyEnv
        for task in ["solar_scheduling", "battery_management", "grid_interaction"]:
            try:
                Q = load_qtable(f"energy_{task}")
                env = EnergyEnv(task=task)
                trained = _eval_encoded(env, Q, EVAL_EPISODES, True)
                random_ = _eval_encoded(env, Q, EVAL_EPISODES, False)
                results.append(("Energy", task, trained, random_))
                print(f"    [OK] energy/{task}: trained={trained:.1f}  random={random_:.1f}")
            except FileNotFoundError:
                print(f"    [SKIP] energy/{task} — Q-table not trained yet")
    except ImportError as e:
        print(f"    [SKIP] energy domain — import error: {e}")

    # ── Finance: custom env API, uses FinanceAgent directly ──────────────────
    try:
        for task in ["trading", "savings", "budget"]:
            try:
                trained = _eval_finance(task, EVAL_EPISODES, True)
                random_ = _eval_finance(task, EVAL_EPISODES, False)
                results.append(("Finance", task, trained, random_))
                print(f"    [OK] finance/{task}: trained={trained:.1f}  random={random_:.1f}")
            except FileNotFoundError:
                print(f"    [SKIP] finance/{task} — Q-table not trained yet")
    except ImportError as e:
        print(f"    [SKIP] finance domain — import error: {e}")

    if not results:
        print("\n  [!] No Q-tables found at all — nothing to plot for the bar chart.")
        print("      Run the training/train_*.py scripts first, then re-run this script.")
        return

    # ── Optionally collapse to domain-level averages ──────────────────────────
    if USE_DOMAIN_AVERAGE:
        by_domain = {}
        for domain, task, trained, random_ in results:
            by_domain.setdefault(domain, {"trained": [], "random": []})
            by_domain[domain]["trained"].append(trained)
            by_domain[domain]["random"].append(random_)
        plot_rows = [
            (domain, "avg of tasks", np.mean(v["trained"]), np.mean(v["random"]))
            for domain, v in by_domain.items()
        ]
    else:
        plot_rows = results

    # Sort by improvement so the poster reads best-story-first
    plot_rows.sort(key=lambda r: (r[2] - r[3]))

    labels  = [f"{d}\n{t.replace('_', ' ').title()}" for d, t, _, _ in plot_rows]
    trained = [r[2] for r in plot_rows]
    random_ = [r[3] for r in plot_rows]

    x = np.arange(len(plot_rows))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(plot_rows) * 1.1), 6))
    ax.bar(x - w/2, trained, w, label="Trained Agent", color="#2E7D32")
    ax.bar(x + w/2, random_, w, label="Random Baseline", color="#B0BEC5")

    for i, (t, r) in enumerate(zip(trained, random_)):
        imp = ((t - r) / max(abs(r), 1)) * 100
        y = max(t, r)
        ax.text(i, y + (abs(y) * 0.03 + 0.5), f"{imp:+.0f}%", ha="center",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Average Episode Reward", fontsize=12)
    ax.set_title("Trained Q-Learning Agent vs. Random Baseline", fontsize=15, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_BAR, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [OK] Saved -> {OUT_BAR}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  UMORDA — Poster Visual Generator")
    print("=" * 65)

    generate_reward_curve()
    generate_trained_vs_random()

    print("\n" + "=" * 65)
    print("  DONE. Files saved in the current directory:")
    print(f"    {OUT_CURVE}")
    print(f"    {OUT_BAR}")
    print("=" * 65 + "\n")
