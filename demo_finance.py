

import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt

# import both environments
sys.path.insert(0, os.path.dirname(__file__))

from environments.finance_env  import FinanceEnv



class QLearningAgent:

    ALPHA     = 0.1
    GAMMA     = 0.95
    EPS_START = 1.0
    EPS_END   = 0.05
    EPS_DECAY = 0.995

    def __init__(self, task, n_actions):
        self.task      = task
        self.n_actions = n_actions

        if task == "trading":
            self.Q = np.zeros((5, 4, 6, n_actions))
        elif task == "savings":
            self.Q = np.zeros((6, 5, 5, n_actions))
        elif task == "budget":
            self.Q = np.zeros((5, 5, 5, n_actions))

        self.eps = self.EPS_START

    def discretise(self, state):
        if self.task == "trading":
            trend  = int(state["price_trend"]) + 2
            shares = min(int(state["shares_held"]), 10) // 3
            cash_buckets=[0,200,500,1000,1500,2000]
            cash = next((i for i, b in enumerate(cash_buckets)
                 if state["cash"] <= b), 5)
            
            return (
                int(np.clip(trend,  0, 4)),
                int(np.clip(shares, 0, 3)),
                int(np.clip(cash,   0, 5)),
            )
        elif self.task == "savings":
            income  = min(int(state["monthly_income"]),  150) // 30
            savings = min(int(state["current_savings"]), 500) // 125
            months  = min(int(state["months_remaining"]), 12) // 3
            return (
                int(np.clip(income,  0, 5)),
                int(np.clip(savings, 0, 4)),
                int(np.clip(months,  0, 4)),
            )
        elif self.task == "budget":
            total    = int(state["total_budget"])
            spent    = int(state["amount_spent"])
            used_pct = min(int((spent / max(total, 1)) * 5), 4)
            urgent   = min(int(state["urgent_requests"]), 4)
            depts    = min(int(state["departments_remaining"]), 4)
            return (
                int(np.clip(used_pct, 0, 4)),
                int(np.clip(urgent,   0, 4)),
                int(np.clip(depts,    0, 4)),
            )

    def choose_action(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        s = self.discretise(state)
        return int(np.argmax(self.Q[s]))

    def update(self, state, action, reward, next_state):
        s  = self.discretise(state)
        s_ = self.discretise(next_state)
        target = reward + self.GAMMA * np.max(self.Q[s_])
        self.Q[s][action] += self.ALPHA * (target - self.Q[s][action])

    def decay_epsilon(self):
        self.eps = max(self.EPS_END, self.eps * self.EPS_DECAY)



# Training loop

def train(env, n_episodes=5000, verbose=True):
    info_env   = env.get_info()
    n_actions  = len(info_env["actions"])
    task       = info_env["task"]           # ← get the task name
    agent      = QLearningAgent(task, n_actions)   # ← pass task
    ep_rewards = []

    for ep in range(n_episodes):
        state   = env.reset()
        total_r = 0.0
        done    = False

        while not done:
            action                    = agent.choose_action(state)
            next_state, r, done, _   = env.step(action)
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



# Evaluation: trained vs random

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



# Plot reward curves

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



# Main
def main():
    EPISODES = 5000
    TICKERS  = ["AAPL", "TSLA", "SPY", "MSFT"]

    tasks = [
        (FinanceEnv, "trading",  "Finance — Trading Agent"),
        (FinanceEnv, "savings",  "Finance — Savings Management"),
        (FinanceEnv, "budget",   "Finance — Budget Allocation"),
    ]

    results = []
    summary = []

    for EnvClass, task, label in tasks:
        print(f"\n{'='*60}")
        print(f"Training: {label}")
        print(f"{'='*60}")

        if task == "trading":
            # Train on all 4 tickers rotating every 1000 episodes
            info_env  = EnvClass(task=task, ticker="AAPL").get_info()
            n_actions = len(info_env["actions"])
            agent     = QLearningAgent(task, n_actions)
            ep_rewards = []

            for ep in range(EPISODES):
                ticker = TICKERS[ep % len(TICKERS)]
                env    = EnvClass(task=task, ticker=ticker)
                state  = env.reset()
                total_r = 0.0
                done    = False

                while not done:
                    action = agent.choose_action(state)
                    next_state, r, done, _ = env.step(action)
                    agent.update(state, action, r, next_state)
                    state   = next_state
                    total_r += r

                agent.decay_epsilon()
                ep_rewards.append(total_r)

                if (ep + 1) % 500 == 0:
                    window = ep_rewards[-500:]
                    print(f"  Episode {ep+1:>5} | "
                          f"avg reward (last 500): {np.mean(window):+.1f} | "
                          f"eps: {agent.eps:.3f}")

            # Evaluate on AAPL
            eval_env = EnvClass(task=task, ticker="AAPL")
            trained, baseline = evaluate(eval_env, agent, n_episodes=200)

        else:
            env = EnvClass(task=task)
            agent, ep_rewards = train(env, n_episodes=EPISODES, verbose=True)
            trained, baseline = evaluate(env, agent, n_episodes=200)

        improvement = ((trained - baseline) / max(abs(baseline), 1)) * 100
        print(f"\n  ✓ Trained agent:  {trained:+.1f}")
        print(f"  ✗ Random agent:   {baseline:+.1f}")
        print(f"  ↑ Improvement:    {improvement:+.1f}%")

        results.append((label, ep_rewards, trained, baseline))
        summary.append((label, trained, baseline, improvement))

    print(f"\n\n{'='*70}")
    print(f"{'FINAL RESULTS SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Task':<35} {'Trained':>10} {'Random':>10} {'Improve%':>10}")
    print(f"{'-'*70}")
    for label, tr, bl, imp in summary:
        print(f"{label:<35} {tr:>+10.1f} {bl:>+10.1f} {imp:>+9.1f}%")
    print(f"{'='*70}")

    plot_all(results, filename="reward_curves.png")


if __name__ == "__main__":
    main()
