

import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display needed -- saves to PNG
import matplotlib.pyplot as plt

# ── import both environments ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from enviroments.finance_env  import FinanceEnv


# ═════════════════════════════════════════════════════════════════════════════
# Q-Learning Agent
# ═════════════════════════════════════════════════════════════════════════════
class QLearningAgent:
    """
    Tabular Q-learning agent.
    State is discretised into BINS buckets per variable so it fits in a table.
    """

    BINS       = 8       # buckets per state variable
    ALPHA      = 0.1     # learning rate
    GAMMA      = 0.95    # discount factor
    EPS_START  = 1.0     # start fully random
    EPS_END    = 0.05    # decay floor
    EPS_DECAY  = 0.995   # multiply eps after each episode

    def __init__(self, n_state_vars, n_actions):
        self.n_actions   = n_actions
        self.n_state_vars = n_state_vars
        # Q-table shape: (BINS, BINS, ..., n_actions) -- one dim per state var
        shape = [self.BINS] * n_state_vars + [n_actions]
        self.Q   = np.zeros(shape)
        self.eps = self.EPS_START

    # ── discretise a raw state dict into Q-table indices ─────────────────────
    def discretise(self, state):
        values = list(state.values())
        indices = []
        for v in values:
            # Clip to [0, 1] based on assumed range, then bucket
            try:
                v_float = float(v)
            except (TypeError, ValueError):
                v_float = 0.0
            # normalise into [0, BINS-1] with soft clipping
            bucket = int(np.clip(abs(v_float) % self.BINS, 0, self.BINS - 1))
            indices.append(bucket)
        return tuple(indices)

    def choose_action(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        s = self.discretise(state)
        return int(np.argmax(self.Q[s]))

    def update(self, state, action, reward, next_state):
        s  = self.discretise(state)
        s_ = self.discretise(next_state)
        target  = reward + self.GAMMA * np.max(self.Q[s_])
        self.Q[s][action] += self.ALPHA * (target - self.Q[s][action])

    def decay_epsilon(self):
        self.eps = max(self.EPS_END, self.eps * self.EPS_DECAY)


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════
def train(env, n_episodes=3000, verbose=True):
    info_env    = env.get_info()
    n_vars      = len(info_env["state_vars"])
    n_actions   = len(info_env["actions"])
    agent       = QLearningAgent(n_vars, n_actions)
    ep_rewards  = []

    for ep in range(n_episodes):
        state      = env.reset()
        total_r    = 0.0
        done       = False

        while not done:
            action               = agent.choose_action(state)
            next_state, r, done, _ = env.step(action)
            agent.update(state, action, r, next_state)
            state    = next_state
            total_r += r

        agent.decay_epsilon()
        ep_rewards.append(total_r)

        if verbose and (ep + 1) % 500 == 0:
            window = ep_rewards[-500:]
            print(f"  Episode {ep+1:>5} | "
                  f"avg reward (last 500): {np.mean(window):+.1f} | "
                  f"eps: {agent.eps:.3f}")

    return agent, ep_rewards


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation: trained vs random
# ═════════════════════════════════════════════════════════════════════════════
def evaluate(env, agent, n_episodes=200):
    """Returns (trained_mean, random_mean)."""
    # Trained
    trained_scores = []
    for _ in range(n_episodes):
        state = env.reset()
        total, done = 0.0, False
        while not done:
            a = agent.choose_action(state, greedy=True)
            state, r, done, _ = env.step(a)
            total += r
        trained_scores.append(total)

    # Random baseline
    random_scores = []
    for _ in range(n_episodes):
        state = env.reset()
        total, done = 0.0, False
        while not done:
            a = random.randint(0, env.n_actions - 1)
            state, r, done, _ = env.step(a)
            total += r
        random_scores.append(total)

    return np.mean(trained_scores), np.mean(random_scores)


# ═════════════════════════════════════════════════════════════════════════════
# Plot reward curves
# ═════════════════════════════════════════════════════════════════════════════
def smooth(arr, window=100):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_all(results, filename="reward_curves.png"):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = np.array(axes).flatten()

    for i, (title, ep_rewards, trained, baseline) in enumerate(results):
        ax = axes[i]
        ax.plot(ep_rewards, alpha=0.15, color="steelblue")
        ax.plot(smooth(ep_rewards), color="steelblue", linewidth=2, label="Smoothed reward")
        ax.axhline(trained,  color="green",  linestyle="--", linewidth=1.2,
                   label=f"Trained avg: {trained:+.1f}")
        ax.axhline(baseline, color="red",    linestyle="--", linewidth=1.2,
                   label=f"Random avg:  {baseline:+.1f}")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("UMORDA — Q-Learning Training Results\n FinanceEnv",
                 fontweight="bold", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nReward curve plot saved → {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    EPISODES = 3000

    tasks = [
        # (EnvClass, task_name, display_label)
        
        (FinanceEnv,  "trading",         "Finance — Trading Agent"),
        (FinanceEnv,  "savings",         "Finance — Savings Management"),
        (FinanceEnv,  "budget",          "Finance — Budget Allocation"),
    ]

    results   = []
    summary   = []

    for EnvClass, task, label in tasks:
        print(f"\n{'='*60}")
        print(f"Training: {label}")
        print(f"{'='*60}")
        env            = EnvClass(task=task)
        agent, rewards = train(env, n_episodes=EPISODES, verbose=True)
        trained, baseline = evaluate(env, agent, n_episodes=200)

        improvement = ((trained - baseline) / max(abs(baseline), 1)) * 100
        print(f"\n  ✓ Trained agent:  {trained:+.1f}")
        print(f"  ✗ Random agent:   {baseline:+.1f}")
        print(f"  ↑ Improvement:    {improvement:+.1f}%")

        results.append((label, rewards, trained, baseline))
        summary.append((label, trained, baseline, improvement))

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"{'FINAL RESULTS SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Task':<35} {'Trained':>10} {'Random':>10} {'Improve%':>10}")
    print(f"{'-'*70}")
    for label, tr, bl, imp in summary:
        print(f"{label:<35} {tr:>+10.1f} {bl:>+10.1f} {imp:>+9.1f}%")
    print(f"{'='*70}")

    # ── Reward curves ─────────────────────────────────────────────────────────
    plot_all(results, filename="reward_curves.png")

    print("\nDone! Files written:")
    print("  → finance_env.py       (Finance environment)")
    print("  → rl_demo_agent.py     (This Q-learning agent)")
    print("  → reward_curves.png    (Training plots)")


if __name__ == "__main__":
    main()
