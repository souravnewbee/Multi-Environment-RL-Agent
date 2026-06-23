"""
UMORDA — Hospital Q-Learning Agent (Gymnasium-compatible)
Pure Q-learning agent with epsilon-greedy exploration, convergence tracking,
and discretization bins matched to the REAL caps used inside hospital_env.py
(previously these were mismatched -- e.g. waiting_patients can reach 60 but
was being clamped to 30 before bucketing, losing resolution exactly where it
matters most: the worst-case states).
"""

import numpy as np
import random
import os


# ── Discretize continuous obs into bins (matches hospital_env.py's real caps) ─
def discretize(obs, task):
    if task == "bed_allocation":
        # free_beds: 0-20 -> 5 bins of 4
        # waiting_patients: 0-60 (real cap) -> 6 bins of 10
        beds     = int(np.clip(obs[0], 0, 20)) // 4
        patients = int(np.clip(obs[1], 0, 60)) // 10
        return (beds, patients)

    elif task == "er_queue":
        # emergency_queue: 0-20 (real cap) -> 5 bins of 4
        # normal_queue: 0-40 (real cap) -> 5 bins of 8
        emergency = int(np.clip(obs[0], 0, 20)) // 4
        normal    = int(np.clip(obs[1], 0, 40)) // 8
        return (emergency, normal)

    elif task == "staff_allocation":
        # available_doctors: 1-20 -> 5 bins of 4
        # patient_load: 0-80 (real cap) -> 5 bins of 16
        doctors = int(np.clip(obs[0], 1, 20)) // 4
        load    = int(np.clip(obs[1], 0, 80)) // 16
        return (doctors, load)

    raise ValueError(f"Unknown task: {task}")


# Q-table shapes matching the bin counts above
Q_SHAPES = {
    "bed_allocation":   (6, 7, 3),    # beds 0-5, patients 0-6
    "er_queue":         (6, 6, 2),    # emergency 0-5, normal 0-5
    "staff_allocation": (6, 6, 3),    # doctors 0-5, load 0-5
}


class HospitalAgent:
    def __init__(
        self,
        task,
        q_shape,
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

        self.Q = np.random.uniform(low=-0.01, high=0.01, size=q_shape)

        self.episode_rewards   = []
        self.episode_epsilons  = []
        self.convergence_delta = []

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        s = discretize(obs, self.task)
        return int(np.argmax(self.Q[s]))

    def update(self, obs, action, reward, next_obs, done):
        s  = discretize(obs,      self.task)
        ns = discretize(next_obs, self.task)

        current_q = self.Q[s][action]
        target_q  = reward + (0.0 if done else self.gamma * np.max(self.Q[ns]))
        old_q     = self.Q[s][action]
        self.Q[s][action] += self.alpha * (target_q - current_q)

        return abs(self.Q[s][action] - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def log_episode(self, total_reward, mean_delta):
        self.episode_rewards.append(total_reward)
        self.episode_epsilons.append(self.epsilon)
        self.convergence_delta.append(mean_delta)

    def is_converged(self, window=1000, stability_ratio=0.05):
        """
        Stochastic environment -> ΔQ never decays to zero, it stabilises
        around a noisy-but-bounded mean. Convergence here means STABILITY:
        compare mean ΔQ of the most recent window against the previous window.
        """
        if len(self.convergence_delta) < window * 2:
            return False
        recent   = np.mean(self.convergence_delta[-window:])
        previous = np.mean(self.convergence_delta[-2 * window:-window])
        if previous == 0:
            return recent == 0
        return abs(recent - previous) / previous < stability_ratio

    def best_action(self, obs):
        s = discretize(obs, self.task)
        return int(np.argmax(self.Q[s]))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.Q)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Q-table not found: {path}")
        self.Q = np.load(path)

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