 """
UMORDA — Agriculture Q-Learning Agent (Gymnasium-compatible)
Pure Q-learning agent with epsilon-greedy exploration, convergence tracking,
and discretization bins matched to the REAL caps used inside agriculture_env.py.

Follows the same structure as hospital_agent.py: proper bin sizes calibrated to
the actual observation ranges (e.g., soil_ph 3.0–9.0, water_reservoir 0–100,
resource_used 0–1000), not arbitrary guesses.

CHANGED: save()/load() now go through qtable_store.py (SQLite) instead of
raw np.save/np.load, so the trained Q-table lives in qtables/qtables.db
alongside every other domain's tables, instead of a standalone .npy file.
"""

import numpy as np
import random
import os

from qtable_store import save_qtable, load_qtable


# ── Discretize continuous obs into bins (matches agriculture_env.py's real caps) ─
def discretize(obs, task):
    if task == "soil_preparation":
        # soil_ph: 3.0–9.0 (6-unit range) -> 4 bins of 1.5
        # organic_matter: 0–100 -> 4 bins of 25
        # drainage_quality: 0–100 -> 4 bins of 25
        # days_remaining: 0–25 -> 3 bins of ~8.3 (approximate)
        ph       = int(np.clip((obs[0] - 3.0) / 1.5, 0, 3))
        om       = int(np.clip(obs[1] / 25, 0, 3))
        drainage = int(np.clip(obs[2] / 25, 0, 3))
        days     = int(np.clip(obs[3] / 8.33, 0, 2))
        return (ph, om, drainage, days)

    elif task == "irrigation":
        # water_reservoir: 0–100 -> 5 bins of 20
        # crop_stress: 0–100 -> 5 bins of 20
        # rainfall_trend: -2 to +2 (4-unit range) -> 5 bins of 0.8
        # days_remaining: 0–40 -> 5 bins of 8
        reservoir = int(np.clip(obs[0] / 20, 0, 4))
        stress    = int(np.clip(obs[1] / 20, 0, 4))
        trend     = int(np.clip((obs[2] + 2) / 0.8, 0, 4))
        days      = int(np.clip(obs[3] / 8, 0, 4))
        return (reservoir, stress, trend, days)

    elif task == "pest_control":
        # total_resource: 0–1000 -> resource budget doesn't discretize directly
        # resource_used: 0–1000 -> 5 bins of 200
        # urgent_outbreaks: 0–10 -> 3 bins of ~3.3
        # plots_remaining: 0–10 -> 3 bins of ~3.3
        # (Note: we discretize by how much of the budget is *used*, not by total)
        used_pct = int(np.clip(obs[1] / 200, 0, 4))
        urgent   = int(np.clip(obs[2] / 3.33, 0, 2))
        plots    = int(np.clip(obs[3] / 3.33, 0, 2))
        return (used_pct, urgent, plots)

    raise ValueError(f"Unknown task: {task}")


# Q-table shapes matching the bin counts above
Q_SHAPES = {
    "soil_preparation": (4, 4, 4, 3, 4),   # ph, om, drainage, days, actions
    "irrigation":       (5, 5, 5, 5, 3),   # reservoir, stress, trend, days, actions
    "pest_control":     (5, 3, 3, 3),      # used_pct, urgent, plots, actions
}


class AgricultureAgent:
    def __init__(
        self,
        task,
        n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
    ):
        self.task          = task
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Build Q-table with the right shape for this task
        q_shape = Q_SHAPES[task] if task in Q_SHAPES else None
        if q_shape is None:
            raise ValueError(f"Unknown task: {task}")
        self.Q = np.random.uniform(low=-0.01, high=0.01, size=q_shape)

        # Tracking for analysis and convergence
        self.episode_rewards   = []
        self.episode_epsilons  = []
        self.convergence_delta = []

    def select_action(self, obs):
        """
        Epsilon-greedy action selection: explore randomly with probability epsilon,
        exploit the best learned action otherwise.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        s = discretize(obs, self.task)
        return int(np.argmax(self.Q[s]))

    def update(self, obs, action, reward, next_obs, done):
        """
        Q-learning update: adjust Q[state, action] toward the observed reward
        plus the discounted future value. Returns the magnitude of the change
        for convergence tracking.
        """
        s  = discretize(obs,      self.task)
        ns = discretize(next_obs, self.task)

        current_q = self.Q[s][action]
        target_q  = reward + (0.0 if done else self.gamma * np.max(self.Q[ns]))
        old_q     = self.Q[s][action]
        self.Q[s][action] += self.alpha * (target_q - current_q)

        return abs(self.Q[s][action] - old_q)

    def decay_epsilon(self):
        """Lower epsilon over time: exploration → exploitation."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def log_episode(self, total_reward, mean_delta):
        """Record episode outcome for analysis and plotting."""
        self.episode_rewards.append(total_reward)
        self.episode_epsilons.append(self.epsilon)
        self.convergence_delta.append(mean_delta)

    def is_converged(self, window=1000, stability_ratio=0.05):
        """
        Convergence for stochastic environments: ΔQ doesn't go to zero, it
        stabilizes around a noisy-but-bounded mean. This checks if the mean
        ΔQ has stopped drifting between recent and prior windows.
        """
        if len(self.convergence_delta) < window * 2:
            return False
        recent   = np.mean(self.convergence_delta[-window:])
        previous = np.mean(self.convergence_delta[-2 * window:-window])
        if previous == 0:
            return recent == 0
        return abs(recent - previous) / previous < stability_ratio

    def best_action(self, obs):
        """Fully greedy action selection (no exploration)."""
        s = discretize(obs, self.task)
        return int(np.argmax(self.Q[s]))

    # ── CHANGED: now saves to / loads from qtables/qtables.db (SQLite) ────────
    # `task` param kept optional for backward compatibility with old call
    # sites that passed a file path — it's ignored now, self.task is used
    # as the DB key so every domain's tables live in one place.
    def save(self, task=None):
        """Persist the Q-table to qtables.db."""
        save_qtable(f"agriculture_{self.task}", self.Q)

    def load(self, task=None):
        """Load a saved Q-table from qtables.db."""
        self.Q = load_qtable(f"agriculture_{self.task}")

    def summary(self):
        """Print training summary stats."""
        if not self.episode_rewards:
            print("  No training data yet.")
            return
        rewards = np.array(self.episode_rewards)
        print(f"  Episodes trained    : {len(rewards)}")
        print(f"  Final epsilon       : {self.epsilon:.4f}")
        print(f"  Avg reward (all)    : {rewards.mean():.2f}")
        print(f"  Avg reward (last 1000) : {rewards[-1000:].mean():.2f}")
        print(f"  Best episode        : {rewards.max():.2f}")
        print(f"  Converged           : {self.is_converged()}")
