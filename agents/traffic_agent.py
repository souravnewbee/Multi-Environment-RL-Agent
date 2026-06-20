# =============================================================================
# UMORDA — Traffic Domain Q-Learning Agent
# File: agents/traffic_agent.py
# =============================================================================

import numpy as np


class TrafficAgent:
    """
    Q-Learning agent for the Traffic domain.

    Bellman update rule:
        Q[s][a] <- Q[s][a] + alpha * (r + gamma * max_a' Q[s'][a'] - Q[s][a])

    Parameters
    ----------
    n_states   : total number of discrete states
    n_actions  : number of possible actions
    alpha      : learning rate          (default 0.1)
    gamma      : discount factor        (default 0.95)
    epsilon    : initial exploration    (default 1.0)
    epsilon_min: minimum exploration    (default 0.01)
    epsilon_decay: decay per episode    (default 0.9995)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-Table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)   # Explore
        return int(np.argmax(self.q_table[state]))     # Exploit

    # ------------------------------------------------------------------
    # Q-Table update (Bellman equation)
    # ------------------------------------------------------------------
    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Apply Bellman equation:
            Q[s][a] <- Q[s][a] + alpha * (r + gamma * max Q[s'][a'] - Q[s][a])
        """
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Best action (exploitation only — no exploration)
    # ------------------------------------------------------------------
    def best_action(self, state: int) -> int:
        """Return the greedy best action for a given state."""
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    # Save / Load Q-Table
    # ------------------------------------------------------------------
    def save_qtable(self, path: str):
        np.save(path, self.q_table)
        print(f"  [✓] Q-Table saved → {path}")

    def load_qtable(self, path: str):
        self.q_table = np.load(path)
        print(f"  [✓] Q-Table loaded ← {path}")

    # ------------------------------------------------------------------
    # Display Q-Table (pretty print)
    # ------------------------------------------------------------------
    def print_qtable(self, action_meanings: list, max_rows: int = 20, title: str = "Q-TABLE"):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        header = f"{'State':>8} | " + " | ".join(f"{a:>18}" for a in action_meanings)
        print(header)
        print("-" * len(header))

        # Only print non-zero rows (or first max_rows)
        printed = 0
        for s in range(self.n_states):
            if np.any(self.q_table[s] != 0) or printed < 5:
                row = f"{s:>8} | " + " | ".join(f"{v:>18.4f}" for v in self.q_table[s])
                print(row)
                printed += 1
                if printed >= max_rows:
                    remaining = sum(1 for i in range(s+1, self.n_states)
                                    if np.any(self.q_table[i] != 0))
                    if remaining:
                        print(f"  ... ({remaining} more non-zero rows)")
                    break
        print(f"{'='*60}\n")
