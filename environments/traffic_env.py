# =============================================================================
# UMORDA — Traffic Domain Environment
# File: environments/traffic_env.py
# Domain: Traffic Management
# Tasks: intersection, pedestrian, parking
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficEnv(gym.Env):
    """
    Multi-task Traffic Management Environment.

    Tasks:
        - intersection : Single traffic intersection control
        - pedestrian   : Pedestrian crossing control
        - parking      : Parking lot management

    Bellman Update (used in agent):
        Q[s][a] <- Q[s][a] + alpha * (r + gamma * max_a' Q[s'][a'] - Q[s][a])
    """

    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Task configuration registry
    # ------------------------------------------------------------------
    TASK_CONFIG = {
        "intersection": {
            "description": "Single traffic intersection control",
            "state_vars": ["cars_N", "cars_S", "cars_E", "cars_W"],
            "state_bins": [5, 5, 5, 5],          # 0-4 cars each direction
            "n_actions": 2,
            "action_meanings": ["Green NS (North-South)", "Green EW (East-West)"],
            "objectives": ["Maximize traffic flow", "Minimize wait time"],
        },
        "pedestrian": {
            "description": "Pedestrian crossing control",
            "state_vars": ["waiting_pedestrians", "waiting_vehicles"],
            "state_bins": [6, 6],                 # 0-5 each
            "n_actions": 2,
            "action_meanings": ["Allow Pedestrians", "Allow Vehicles"],
            "objectives": ["Maximize safety", "Minimize wait time"],
        },
        "parking": {
            "description": "Parking lot management",
            "state_vars": ["available_spots", "incoming_vehicles"],
            "state_bins": [11, 6],                # 0-10 spots, 0-5 vehicles
            "n_actions": 3,
            "action_meanings": ["Open Zone A", "Open Zone B", "Close Entry"],
            "objectives": ["Maximize occupancy", "Minimize congestion"],
        },
    }

    def __init__(self, task: str = "intersection", render_mode=None):
        super().__init__()

        assert task in self.TASK_CONFIG, (
            f"Unknown task '{task}'. Choose from: {list(self.TASK_CONFIG.keys())}"
        )

        self.task = task
        self.render_mode = render_mode
        self.cfg = self.TASK_CONFIG[task]

        # Observation & action spaces
        state_size = int(np.prod(self.cfg["state_bins"]))
        self.observation_space = spaces.Discrete(state_size)
        self.action_space = spaces.Discrete(self.cfg["n_actions"])

        self._state = None
        self._raw_state = None
        self._step_count = 0
        self.max_steps = 100

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_state(self, raw: list) -> int:
        """Convert multi-dimensional raw state to a single integer index."""
        bins = self.cfg["state_bins"]
        idx = 0
        multiplier = 1
        for i in reversed(range(len(raw))):
            idx += raw[i] * multiplier
            multiplier *= bins[i]
        return idx

    def _sample_raw_state(self) -> list:
        """Sample a random raw state within bins."""
        return [np.random.randint(0, b) for b in self.cfg["state_bins"]]

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._raw_state = self._sample_raw_state()
        self._state = self._encode_state(self._raw_state)
        self._step_count = 0
        return self._state, {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward = self._compute_reward(action)
        self._raw_state = self._next_raw_state(action)
        self._state = self._encode_state(self._raw_state)
        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        truncated = False

        if self.render_mode == "human":
            self.render()

        return self._state, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Reward logic (per task)
    # ------------------------------------------------------------------
    def _compute_reward(self, action: int) -> float:

        if self.task == "intersection":
            cars_N, cars_S, cars_E, cars_W = self._raw_state
            ns_total = cars_N + cars_S
            ew_total = cars_E + cars_W
            if action == 0:   # Green NS
                reward = ns_total - 0.5 * ew_total
            else:             # Green EW
                reward = ew_total - 0.5 * ns_total
            reward = float(np.clip(reward, -10, 10))

        elif self.task == "pedestrian":
            waiting_peds, waiting_vehs = self._raw_state
            if action == 0:   # Allow Pedestrians
                reward = 2.0 * waiting_peds - 0.5 * waiting_vehs
                if waiting_peds == 0:
                    reward -= 3.0   # penalty: no one to cross
            else:             # Allow Vehicles
                reward = 1.5 * waiting_vehs - 1.0 * waiting_peds
                if waiting_peds > 3:
                    reward -= 4.0   # safety penalty
            reward = float(np.clip(reward, -10, 10))

        elif self.task == "parking":
            available_spots, incoming_vehicles = self._raw_state
            if action == 0:   # Open Zone A
                reward = min(incoming_vehicles, available_spots) * 2.0
                reward -= max(0, incoming_vehicles - available_spots) * 1.0
            elif action == 1: # Open Zone B
                reward = min(incoming_vehicles, available_spots) * 1.5
                reward -= max(0, incoming_vehicles - available_spots) * 0.5
            else:             # Close Entry
                if available_spots < 2:
                    reward = 3.0    # good: lot is full
                else:
                    reward = -2.0   # bad: closed unnecessarily
            reward = float(np.clip(reward, -10, 10))

        else:
            reward = 0.0

        return reward

    # ------------------------------------------------------------------
    # State transition (simulated next state)
    # ------------------------------------------------------------------
    def _next_raw_state(self, action: int) -> list:
        raw = self._raw_state.copy()
        bins = self.cfg["state_bins"]

        if self.task == "intersection":
            if action == 0:   # Green NS → reduce N/S, increase E/W slightly
                raw[0] = max(0, raw[0] - np.random.randint(1, 3))
                raw[1] = max(0, raw[1] - np.random.randint(1, 3))
                raw[2] = min(bins[2]-1, raw[2] + np.random.randint(0, 2))
                raw[3] = min(bins[3]-1, raw[3] + np.random.randint(0, 2))
            else:             # Green EW
                raw[2] = max(0, raw[2] - np.random.randint(1, 3))
                raw[3] = max(0, raw[3] - np.random.randint(1, 3))
                raw[0] = min(bins[0]-1, raw[0] + np.random.randint(0, 2))
                raw[1] = min(bins[1]-1, raw[1] + np.random.randint(0, 2))
            # Random new arrivals
            for i in range(4):
                raw[i] = min(bins[i]-1, raw[i] + np.random.randint(0, 2))

        elif self.task == "pedestrian":
            if action == 0:   # Allow Pedestrians
                raw[0] = max(0, raw[0] - np.random.randint(1, 3))
                raw[1] = min(bins[1]-1, raw[1] + np.random.randint(0, 2))
            else:             # Allow Vehicles
                raw[1] = max(0, raw[1] - np.random.randint(1, 3))
                raw[0] = min(bins[0]-1, raw[0] + np.random.randint(0, 2))

        elif self.task == "parking":
            if action == 0:   # Open Zone A
                raw[0] = max(0, raw[0] - min(raw[1], np.random.randint(1, 3)))
                raw[1] = max(0, raw[1] - np.random.randint(1, 2))
            elif action == 1: # Open Zone B
                raw[0] = max(0, raw[0] - min(raw[1], np.random.randint(0, 2)))
                raw[1] = max(0, raw[1] - np.random.randint(0, 2))
            else:             # Close Entry
                raw[1] = max(0, raw[1] - np.random.randint(0, 2))
            raw[1] = min(bins[1]-1, raw[1] + np.random.randint(0, 2))

        return raw

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        cfg = self.cfg
        print(f"\n[Traffic-{self.task.upper()}] Step {self._step_count}")
        for var, val in zip(cfg["state_vars"], self._raw_state):
            print(f"  {var}: {val}")

    def get_state_info(self):
        """Return current raw state as a labeled dict."""
        return dict(zip(self.cfg["state_vars"], self._raw_state))
