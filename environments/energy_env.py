# =============================================================================
# UMORDA — Energy Domain Environment
# File: environments/energy_env.py
#
# Domain: Balcony Solar Panel (Balkonkraftwerk) Optimization
#
# DATASET INTEGRATION (3 steps as per plan):
#   Step 1 — Load CSV once when environment starts
#   Step 2 — reset() picks a random row from real data
#   Step 3 — _next_raw_state() moves to next row in real data
#
# Data files (run fetch scripts first):
#   data/energy/solar_data.csv    ← from NASA POWER API
#   data/energy/grid_price.csv    ← from BPDB tariff slabs
#
# If CSV files are missing → falls back to random (nothing breaks)
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

# ── CSV paths ──────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "energy")
SOLAR_CSV = os.path.join(_BASE, "solar_data.csv")
PRICE_CSV = os.path.join(_BASE, "grid_price.csv")


class EnergyEnv(gym.Env):
    """
    Multi-task Balcony Solar Panel Optimization Environment.

    Tasks:
        solar_scheduling   — how to use solar power right now
        battery_management — when to charge/discharge battery
        grid_interaction   — when to buy/sell grid electricity

    Real data:
        solar_output → NASA POWER hourly irradiance for Dhaka
        grid_price   → BPDB Bangladesh electricity tariff
    """

    metadata = {"render_modes": ["human"]}

    TASK_CONFIG = {
        "solar_scheduling": {
            "description": "Balcony solar panel power scheduling",
            "state_vars":  ["solar_output", "home_consumption",
                            "battery_level", "time_of_day"],
            "state_bins":  [10, 10, 10, 4],
            "n_actions":   3,
            "action_meanings": [
                "Use Solar Directly",
                "Store in Battery",
                "Buy from Grid",
            ],
            "objectives": ["Maximize solar usage", "Minimize grid dependency"],
        },
        "battery_management": {
            "description": "Battery storage charge/discharge optimization",
            "state_vars":  ["battery_level", "solar_output",
                            "grid_price", "home_consumption"],
            "state_bins":  [10, 10, 3, 10],
            "n_actions":   3,
            "action_meanings": [
                "Charge Battery",
                "Discharge Battery",
                "Keep Battery Idle",
            ],
            "objectives": ["Maximize battery efficiency", "Minimize electricity cost"],
        },
        "grid_interaction": {
            "description": "Grid energy buying and selling optimization",
            "state_vars":  ["grid_price", "solar_surplus",
                            "battery_level", "home_consumption"],
            "state_bins":  [3, 10, 10, 10],
            "n_actions":   3,
            "action_meanings": [
                "Buy from Grid",
                "Sell to Grid",
                "Stay Self-Sufficient",
            ],
            "objectives": ["Minimize electricity cost", "Maximize earnings from surplus"],
        },
    }

    def __init__(self, task="solar_scheduling", render_mode=None):
        super().__init__()
        assert task in self.TASK_CONFIG, \
            f"Unknown task '{task}'. Choose from: {list(self.TASK_CONFIG.keys())}"

        self.task        = task
        self.render_mode = render_mode
        self.cfg         = self.TASK_CONFIG[task]

        state_size = int(np.prod(self.cfg["state_bins"]))
        self.observation_space = spaces.Discrete(state_size)
        self.action_space      = spaces.Discrete(self.cfg["n_actions"])

        self._raw_state       = None
        self._state           = None
        self._step_count      = 0
        self.max_steps        = 100
        self._data_idx        = 0
        self._using_real_data = False

        # Real data arrays (loaded from CSV)
        self._solar_rows = []   # list of (solar_scaled, time_of_day)
        self._price_map  = {}   # hour → price_level

        # ── STEP 1: Load CSV files once ───────────────────────────────────
        self._load_csv_data()

    # ------------------------------------------------------------------
    # STEP 1 — Load CSV files once when environment starts
    # ------------------------------------------------------------------
    def _load_csv_data(self):
        """
        Reads solar_data.csv and grid_price.csv into memory once.
        If files are missing, silently falls back to random behavior.
        """
        try:
            import pandas as pd

            solar_ok = False
            price_ok = False

            # Load NASA POWER solar data
            if os.path.exists(SOLAR_CSV):
                df = pd.read_csv(SOLAR_CSV)
                # Store only what we need: (solar_scaled, time_of_day)
                self._solar_rows = list(
                    zip(df["solar_output_scaled"].astype(int),
                        df["time_of_day"].astype(int))
                )
                solar_ok = True

            # Load BPDB grid prices
            if os.path.exists(PRICE_CSV):
                df = pd.read_csv(PRICE_CSV)
                for _, row in df.iterrows():
                    self._price_map[int(row["hour"])] = int(row["price_level"])
                price_ok = True

            if solar_ok and price_ok:
                self._using_real_data = True

        except Exception:
            self._using_real_data = False

    # ------------------------------------------------------------------
    # State encoding / decoding
    # ------------------------------------------------------------------
    def _encode_state(self, raw: list) -> int:
        bins = self.cfg["state_bins"]
        idx, multiplier = 0, 1
        for i in reversed(range(len(raw))):
            idx += int(np.clip(raw[i], 0, bins[i]-1)) * multiplier
            multiplier *= bins[i]
        return idx

    def decode_state(self, state_idx: int) -> list:
        bins, values = self.cfg["state_bins"], []
        for b in reversed(bins):
            values.append(state_idx % b)
            state_idx //= b
        return list(reversed(values))

    def _sample_random_state(self) -> list:
        """Random fallback state when CSV not available."""
        return [np.random.randint(0, b) for b in self.cfg["state_bins"]]

    # ------------------------------------------------------------------
    # STEP 2 — reset() picks a random row from real data
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        if self._using_real_data:
            # Pick a random starting row from the CSV data
            max_start      = max(1, len(self._solar_rows) - self.max_steps - 1)
            self._data_idx = np.random.randint(0, max_start)
            self._raw_state = self._build_state_from_csv(self._data_idx)
        else:
            self._raw_state = self._sample_random_state()

        self._state = self._encode_state(self._raw_state)
        return self._state, {}

    def _build_state_from_csv(self, idx: int) -> list:
        """Build a state vector from real CSV data at given row index."""
        row         = self._solar_rows[idx % len(self._solar_rows)]
        solar       = int(row[0])          # already 0-9 scaled
        time_of_day = int(row[1])

        # Map time_of_day to representative hour for BPDB lookup
        hour_map   = {0: 9, 1: 14, 2: 18, 3: 23}
        hour       = hour_map.get(time_of_day, 12)
        grid_price = self._price_map.get(hour, 1)

        # Realistic home consumption by time of day
        base = {0: 4, 1: 3, 2: 6, 3: 2}
        home_consumption = int(np.clip(
            base[time_of_day] + np.random.randint(-1, 2), 0, 9
        ))
        battery_level = np.random.randint(2, 8)

        if self.task == "solar_scheduling":
            return [solar, home_consumption, battery_level, time_of_day]
        elif self.task == "battery_management":
            return [battery_level, solar, grid_price, home_consumption]
        elif self.task == "grid_interaction":
            surplus = max(0, solar - home_consumption)
            return [grid_price, surplus, battery_level, home_consumption]

        return self._sample_random_state()

    # ------------------------------------------------------------------
    # Gymnasium step
    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self.action_space.contains(action)
        reward          = self._compute_reward(action)
        self._raw_state = self._next_raw_state(action)
        self._state     = self._encode_state(self._raw_state)
        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        return self._state, reward, terminated, False, {}

    # ------------------------------------------------------------------
    # STEP 3 — _next_raw_state() moves to next row in real data
    # ------------------------------------------------------------------
    def _next_raw_state(self, action: int) -> list:
        if self._using_real_data:
            # Move to next row in the real CSV data
            self._data_idx += 1
            next_state = self._build_state_from_csv(self._data_idx)
            # Apply action effects on top of the real data transition
            return self._apply_action_effects(action, next_state)
        else:
            return self._next_random_state(action)

    def _apply_action_effects(self, action: int, state: list) -> list:
        """Apply the agent's action effects on the next real data state."""
        bins = self.cfg["state_bins"]

        if self.task == "solar_scheduling":
            solar, consumption, battery, time = state
            if action == 1:   # Store in Battery
                battery = min(bins[2]-1, battery + min(solar, 2))
            elif action == 0: # Use Solar Directly
                solar = max(0, solar - 1)
            return [solar, consumption, battery, time]

        elif self.task == "battery_management":
            battery, solar, price, consumption = state
            if action == 0:   battery = min(bins[0]-1, battery + 1)   # Charge
            elif action == 1: battery = max(0, battery - 1)           # Discharge
            return [battery, solar, price, consumption]

        elif self.task == "grid_interaction":
            price, surplus, battery, consumption = state
            if action == 1:   surplus = max(0, surplus - 1)   # Sell
            elif action == 2: battery = max(0, battery - 1)   # Self-sufficient
            return [price, surplus, battery, consumption]

        return state

    def _next_random_state(self, action: int) -> list:
        """Random state transition fallback."""
        raw  = self._raw_state.copy()
        bins = self.cfg["state_bins"]

        if self.task == "solar_scheduling":
            solar, consumption, battery, time = raw
            if action == 0:
                solar       = max(0, solar - np.random.randint(1, 3))
                consumption = max(0, consumption - np.random.randint(1, 3))
            elif action == 1:
                battery = min(bins[2]-1, battery + min(solar, np.random.randint(1, 3)))
                solar   = max(0, solar - np.random.randint(1, 3))
            else:
                consumption = max(0, consumption - np.random.randint(1, 3))
            time = (time + np.random.randint(0, 2)) % 4
            if time == 1:   solar = min(bins[0]-1, solar + np.random.randint(1, 4))
            elif time == 0: solar = min(bins[0]-1, solar + np.random.randint(0, 3))
            else:           solar = max(0, solar - np.random.randint(0, 3))
            consumption = min(bins[1]-1, consumption + np.random.randint(0, 3))
            raw = [solar, consumption, battery, time]

        elif self.task == "battery_management":
            battery, solar, price, consumption = raw
            if action == 0:
                battery = min(bins[0]-1, battery + np.random.randint(1, 3))
                solar   = max(0, solar - np.random.randint(0, 2))
            elif action == 1:
                battery     = max(0, battery - np.random.randint(1, 3))
                consumption = max(0, consumption - np.random.randint(1, 3))
            price       = int(np.clip(price + np.random.randint(-1, 2), 0, 2))
            solar       = min(bins[1]-1, max(0, solar + np.random.randint(-2, 3)))
            consumption = min(bins[3]-1, max(0, consumption + np.random.randint(-1, 2)))
            raw = [battery, solar, price, consumption]

        elif self.task == "grid_interaction":
            price, surplus, battery, consumption = raw
            if action == 0:   consumption = max(0, consumption - np.random.randint(1, 3))
            elif action == 1: surplus = max(0, surplus - np.random.randint(1, 3))
            else:
                battery = max(0, battery - np.random.randint(0, 2))
                surplus = max(0, surplus - np.random.randint(0, 2))
            price   = int(np.clip(price + np.random.randint(-1, 2), 0, 2))
            surplus = min(bins[1]-1, max(0, surplus + np.random.randint(0, 3)))
            battery = min(bins[2]-1, max(0, battery + np.random.randint(-1, 2)))
            consumption = min(bins[3]-1, max(0, consumption + np.random.randint(-1, 2)))
            raw = [price, surplus, battery, consumption]

        return [int(np.clip(raw[i], 0, self.cfg["state_bins"][i]-1))
                for i in range(len(raw))]

    # ------------------------------------------------------------------
    # Reward logic
    # ------------------------------------------------------------------
    def _compute_reward(self, action: int) -> float:
        if self.task == "solar_scheduling":
            solar, consumption, battery, time = self._raw_state
            if action == 0:
                reward = min(solar, consumption) * 2.0
                if solar == 0:          reward -= 4.0
                if solar > consumption: reward += (solar - consumption) * 0.5
            elif action == 1:
                if solar > 0 and battery < 9: reward = solar * 1.5 - (battery / 9) * 2.0
                elif battery >= 9:             reward = -3.0
                else:                          reward = -2.0
            else:
                if solar == 0 and battery == 0: reward = 2.0
                elif solar > 0:                 reward = -3.0
                else:                           reward = 0.5
            return float(np.clip(reward, -10, 10))

        elif self.task == "battery_management":
            battery, solar, price, consumption = self._raw_state
            if action == 0:
                if solar > 3 and battery < 8:    reward = solar * 1.5 + (1 - price) * 2.0
                elif price == 0 and battery < 8: reward = 2.0
                elif battery >= 8:               reward = -3.0
                else:                            reward = 0.5
            elif action == 1:
                if battery > 2 and price == 2:        reward = battery * 1.5 + consumption * 0.5
                elif battery > 2 and consumption > 5: reward = battery * 1.0
                elif battery <= 2:                    reward = -4.0
                else:                                 reward = 0.0
            else:
                if solar >= consumption: reward = 2.0
                elif price == 1:         reward = 1.0
                else:                    reward = -1.0
            return float(np.clip(reward, -10, 10))

        elif self.task == "grid_interaction":
            price, surplus, battery, consumption = self._raw_state
            if action == 0:
                if battery == 0 and surplus == 0: reward = 3.0 - price * 2.0
                elif surplus > 0 or battery > 3:  reward = -3.0
                else:                             reward = 1.0 - price * 1.5
            elif action == 1:
                if surplus > 3:    reward = surplus * 1.5 + price * 2.0
                elif surplus == 0: reward = -4.0
                else:              reward = surplus * 0.8
            else:
                own = surplus + battery
                if own >= consumption: reward = 4.0 + price * 1.0
                elif own > 0:          reward = own * 0.5
                else:                  reward = -2.0
            return float(np.clip(reward, -10, 10))

        return 0.0

    def render(self):
        mode = "REAL-DATA" if self._using_real_data else "RANDOM"
        print(f"\n[Energy-{self.task.upper()} | {mode}] Step {self._step_count}")
        for var, val in zip(self.cfg["state_vars"], self._raw_state):
            print(f"  {var}: {val}")

    def get_state_info(self):
        return dict(zip(self.cfg["state_vars"], self._raw_state))
