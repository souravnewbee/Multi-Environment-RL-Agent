"""
UMORDA — Hospital Q-Learning Agent
Pure Q-learning agent with:
  - Epsilon-greedy exploration with exponential decay
  - Convergence tracking (checks if Q-table stopped changing meaningfully)
  - Per-episode reward logging for learning curves
  - Supports save / load of Q-tables
"""

import numpy as np
import random
import os


# ── Discretize continuous obs into bins ─────────────────────────────────────
def discretize(obs, task):
    """Map raw observation array → tuple index into Q-table."""
    if task == "bed_allocation":
        beds     = int(np.clip(obs[0], 0, 20)) // 5          # bins: 0-4  (5 bins)
        patients = int(np.clip(obs[1], 0, 30)) // 10         # bins: 0-3  (4 bins)
        return (beds, patients)

    elif task == "er_queue":
        emergency = int(np.clip(obs[0], 0, 10)) // 3         # bins: 0-3  (4 bins)
        normal    = int(np.clip(obs[1], 0, 20)) // 5         # bins: 0-4  (5 bins)
        return (emergency, normal)

    elif task == "staff_allocation":
        doctors = int(np.clip(obs[0], 1, 15)) // 5           # bins: 0-3  (4 bins)
        load    = int(np.clip(obs[1], 0, 50)) // 15          # bins: 0-3  (4 bins)
        return (doctors, load)

    raise ValueError(f"Unknown task: {task}")


class HospitalAgent:
    """
    Tabular Q-learning agent for UMORDA hospital tasks.

    Hyperparameters
    ---------------
    alpha        : learning rate
    gamma        : discount factor
    epsilon      : initial exploration rate (1.0 = fully random)
    epsilon_min  : floor for epsilon
    epsilon_decay: multiplicative decay applied after every episode
    q_shape      : shape of Q-table (state_dims... , n_actions)
    task         : which hospital sub-task this agent handles
    """

    def __init__(
        self,
        task,
        q_shape,
        n_actions,
        alpha=0.1,
        gamma=0.97,
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

        # Q-table initialised to small random values to break ties
        self.Q = np.random.uniform(low=-0.01, high=0.01, size=q_shape)

        # Logging
        self.episode_rewards   = []   # total reward per episode
        self.episode_epsilons  = []   # epsilon value per episode
        self.convergence_delta = []   # mean |Q change| per episode

    # ── Action selection ─────────────────────────────────────────────────────
    def select_action(self, obs):
        """Epsilon-greedy: explore randomly or exploit best known action."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        s = discretize(obs, self.task)
        return int(np.argmax(self.Q[s]))

    # ── Q-table update (single transition) ───────────────────────────────────
    def update(self, obs, action, reward, next_obs, done):
        """Standard Q-learning (off-policy TD update)."""
        s  = discretize(obs,      self.task)
        ns = discretize(next_obs, self.task)

        current_q  = self.Q[s][action]
        target_q   = reward + (0.0 if done else self.gamma * np.max(self.Q[ns]))
        td_error   = target_q - current_q
        old_q      = self.Q[s][action]

        self.Q[s][action] += self.alpha * td_error

        return abs(self.Q[s][action] - old_q)   # return magnitude of change

    # ── Epsilon decay (call once per episode) ────────────────────────────────
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Log episode ──────────────────────────────────────────────────────────
    def log_episode(self, total_reward, mean_delta):
        self.episode_rewards.append(total_reward)
        self.episode_epsilons.append(self.epsilon)
        self.convergence_delta.append(mean_delta)

    # ── Convergence check ────────────────────────────────────────────────────
    def is_converged(self, window=500, threshold=0.001):
        """
        Returns True if the mean Q-change over the last `window` episodes
        has dropped below `threshold` — i.e. the agent has stabilised.
        """
        if len(self.convergence_delta) < window:
            return False
        recent = self.convergence_delta[-window:]
        return float(np.mean(recent)) < threshold

    # ── Best action (greedy, no exploration) ─────────────────────────────────
    def best_action(self, obs):
        s = discretize(obs, self.task)
        return int(np.argmax(self.Q[s]))

    # ── Save / Load ──────────────────────────────────────────────────────────
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.Q)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Q-table not found: {path}")
        self.Q = np.load(path)

    # ── Summary ──────────────────────────────────────────────────────────────
    def summary(self):
        if not self.episode_rewards:
            print("  No training data yet.")
            return
        rewards = np.array(self.episode_rewards)
        print(f"  Episodes trained : {len(rewards)}")
        print(f"  Final epsilon    : {self.epsilon:.4f}")
        print(f"  Avg reward (all) : {rewards.mean():.2f}")
        print(f"  Avg reward (last 1000) : {rewards[-1000:].mean():.2f}")
        print(f"  Best episode     : {rewards.max():.2f}")
        print(f"  Converged        : {self.is_converged()}")
