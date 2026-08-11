# =============================================================================
# UMORDA — Traffic Domain Q-Learning Agent (FIXED VERSION v2)
# File: agents/traffic_agent.py
#
# CHANGED: save_qtable()/load_qtable() now go through qtable_store.py
# (SQLite) instead of raw np.save/np.load. Callers now pass a NAME
# (e.g. "traffic_intersection") instead of a file path.
# =============================================================================

import numpy as np

from qtable_store import save_qtable as _db_save_qtable, load_qtable as _db_load_qtable


class TrafficAgent:
    """
    Q-Learning agent for the Traffic domain.

    Bellman update rule:
        Q[s][a] <- Q[s][a] + alpha * (r + gamma * max_a' Q[s'][a'] - Q[s][a])

    Improvements in v2:
    - Works with larger state spaces (intersection, pedestrian, parking all fixed)
    - Safety override is handled at environment level (hard guarantee)
    - Optimistic initialization helps explore safety-critical states
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
        optimistic_init: float = 1.0,   # slight optimism encourages exploration
    ):
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Optimistic initialization — encourages visiting all states
        self.q_table = np.full((n_states, n_actions), optimistic_init)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Bellman equation update."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state: int) -> int:
        return int(np.argmax(self.q_table[state]))

    def save_qtable(self, name: str):
        """
        `name` should be a DB key like 'traffic_intersection'
        (previously this was a file path like 'qtables/traffic_x_qtable.npy').
        """
        _db_save_qtable(name, self.q_table)
        print(f"  [✓] Q-Table saved → qtables.db (as '{name}')")

    def load_qtable(self, name: str):
        self.q_table = _db_load_qtable(name)
        print(f"  [✓] Q-Table loaded ← qtables.db (as '{name}')")

    def print_qtable(self, action_meanings: list, max_rows: int = 20, title: str = "Q-TABLE"):
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
        header = f"{'State':>8} | " + " | ".join(f"{a:>20}" for a in action_meanings)
        print(header)
        print("-" * len(header))
        printed = 0
        for s in range(self.n_states):
            if np.any(self.q_table[s] != 1.0) or printed < 3:
                row = f"{s:>8} | " + " | ".join(f"{v:>20.4f}" for v in self.q_table[s])
                print(row)
                printed += 1
                if printed >= max_rows:
                    print(f"  ... (more rows exist)")
                    break
        print(f"{'='*65}\n")