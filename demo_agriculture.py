
import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import the environment
sys.path.insert(0, os.path.dirname(__file__))

from environments.agriculture_env import AgricultureEnv


class QLearningAgent:

    ALPHA     = 0.1
    GAMMA     = 0.95
    EPS_START = 1.0
    EPS_END   = 0.05
    EPS_DECAY = 0.995

    def __init__(self, task, n_actions):
        self.task      = task
        self.n_actions = n_actions

        # Q-table shapes match the discretisation buckets used below.
        if task == "soil_preparation":
            # (ph_bucket, organic_matter_bucket, drainage_bucket, days_bucket)
            self.Q = np.zeros((3, 3, 3, 3, n_actions))
        elif task == "irrigation":
            # (reservoir_bucket, stress_bucket, rainfall_trend_bucket)
            self.Q = np.zeros((4, 4, 5, n_actions))
        elif task == "pest_control":
            # (used_pct_bucket, urgent_bucket, plots_bucket) -- same shape as
            # FinanceEnv's "budget" task, since it's the same underlying pattern
            self.Q = np.zeros((5, 5, 5, n_actions))

        self.eps = self.EPS_START

    def discretise(self, obs):
        """obs is the raw numpy array AgricultureEnv._obs() returns, in the
        order given by each task's state_vars."""
        if self.task == "soil_preparation":
            ph, om, drainage, days = obs

            ph_bucket = 0 if ph < 5.5 else (1 if ph <= 7.0 else 2)
            om_bucket = 0 if om < 30 else (1 if om < 60 else 2)
            drain_bucket = 0 if drainage < 30 else (1 if drainage < 60 else 2)
            days_bucket = 0 if days > 15 else (1 if days > 5 else 2)

            return (
                int(np.clip(ph_bucket, 0, 2)),
                int(np.clip(om_bucket, 0, 2)),
                int(np.clip(drain_bucket, 0, 2)),
                int(np.clip(days_bucket, 0, 2)),
            )

        elif self.task == "irrigation":
            reservoir, stress, trend, _days = obs

            reservoir_bucket = min(int(reservoir // 25), 3)
            stress_bucket    = min(int(stress // 25), 3)
            trend_bucket     = int(np.clip(trend + 2, 0, 4))

            return (
                int(np.clip(reservoir_bucket, 0, 3)),
                int(np.clip(stress_bucket, 0, 3)),
                trend_bucket,
            )

        elif self.task == "pest_control":
            total, used, urgent, plots = obs

            used_pct     = min(int((used / max(total, 1)) * 5), 4)
            urgent_bucket = min(int(urgent), 4)
            plots_bucket  = min(int(plots), 4)

            return (
                int(np.clip(used_pct, 0, 4)),
                int(np.clip(urgent_bucket, 0, 4)),
                int(np.clip(plots_bucket, 0, 4)),
            )

    def choose_action(self, obs, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        s = self.discretise(obs)
        return int(np.argmax(self.Q[s]))

    def update(self, obs, action, reward, next_obs):
        s  = self.discretise(obs)
        s_ = self.discretise(next_obs)
        target = reward + self.GAMMA * np.max(self.Q[s_])
        self.Q[s][action] += self.ALPHA * (target - self.Q[s][action])

    def decay_epsilon(self):
        self.eps = max(self.EPS_END, self.eps * self.EPS_DECAY)


# Training loop

def train(env, n_episodes=5000, verbose=True):
    info_env  = env.get_info()
    n_actions = len(info_env["actions"])
    task      = info_env["task"]
    agent     = QLearningAgent(task, n_actions)
    ep_rewards = []

    for ep in range(n_episodes):
        obs, _info = env.reset()
        total_r = 0.0
        done    = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, r, terminated, truncated, _info = env.step(action)
            agent.update(obs, action, r, next_obs)
            obs      = next_obs
            total_r += r
            done     = terminated or truncated

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
        obs, _info = env.reset()
        total, done = 0.0, False
        while not done:
            a = agent.choose_action(obs, greedy=True)
            obs, r, terminated, truncated, _info = env.step(a)
            total += r
            done   = terminated or truncated
        trained_scores.append(total)

    # Random baseline
    random_scores = []
    for _ in range(n_episodes):
        obs, _info = env.reset()
        total, done = 0.0, False
        while not done:
            a = random.randint(0, env.n_actions - 1)
            obs, r, terminated, truncated, _info = env.step(a)
            total += r
            done   = terminated or truncated
        random_scores.append(total)

    return np.mean(trained_scores), np.mean(random_scores)


# Plot reward curves

def smooth(arr, window=100):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_all(results, filename="reward_curves_agriculture.png"):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = np.array(axes).flatten()

    for i, (title, ep_rewards, trained, baseline) in enumerate(results):
        ax = axes[i]
        ax.plot(ep_rewards, alpha=0.15, color="seagreen")
        ax.plot(smooth(ep_rewards), color="seagreen", linewidth=2, label="Smoothed reward")
        ax.axhline(trained,  color="green",  linestyle="--", linewidth=1.2,
                   label=f"Trained avg: {trained:+.1f}")
        ax.axhline(baseline, color="red",    linestyle="--", linewidth=1.2,
                   label=f"Random avg:  {baseline:+.1f}")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("UMORDA — Q-Learning Training Results\n AgricultureEnv",
                 fontweight="bold", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nReward curve plot saved -> {filename}")


# Main
def main():
    EPISODES = 5000

    tasks = [
        (AgricultureEnv, "soil_preparation", "Agriculture — Soil Preparation"),
        (AgricultureEnv, "irrigation",       "Agriculture — Irrigation Management"),
        (AgricultureEnv, "pest_control",     "Agriculture — Pest Control"),
    ]

    results = []
    summary = []

    for EnvClass, task, label in tasks:
        print(f"\n{'='*60}")
        print(f"Training: {label}")
        print(f"{'='*60}")

        env = EnvClass(task=task)
        agent, ep_rewards = train(env, n_episodes=EPISODES, verbose=True)
        trained, baseline = evaluate(env, agent, n_episodes=200)

        improvement = ((trained - baseline) / max(abs(baseline), 1)) * 100
        print(f"\n  Trained agent:  {trained:+.1f}")
        print(f"  Random agent:   {baseline:+.1f}")
        print(f"  Improvement:    {improvement:+.1f}%")

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

    plot_all(results, filename="reward_curves_agriculture.png")


if __name__ == "__main__":
    main()
