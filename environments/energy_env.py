# =============================================================================
# UMORDA — Energy Domain Environment
# File: environments/energy_env.py
#
# Domain: Balcony Solar Panel (Balkonkraftwerk) Optimization
# Inspired by: German balcony solar panel systems
#
# Tasks:
#   solar_scheduling   — decide how to use solar power being generated
#   battery_management — decide when to charge/discharge the battery
#   grid_interaction   — decide when to buy from or sell to the grid
#
# Bellman Update:
#   Q[s][a] <- Q[s][a] + alpha * (r + gamma * max Q[s'][a'] - Q[s][a])
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EnergyEnv(gym.Env):
    """
    Multi-task Balcony Solar Panel Optimization Environment.

    Real-world inspiration: Balkonkraftwerk (Balcony Power Station)
    These are small solar panels (300-800W) used in German apartments.
    The agent learns to optimize energy usage, battery storage, and
    grid interaction to minimize electricity costs.
    """

    metadata = {"render_modes": ["human"]}

    TASK_CONFIG = {

        # ------------------------------------------------------------------
        # TASK 1: Solar Scheduling
        # Decides HOW to use the solar power being generated right now
        # ------------------------------------------------------------------
        "solar_scheduling": {
            "description": "Balcony solar panel power scheduling",
            "state_vars": [
                "solar_output",      # how much solar is generating (0-9: 0=none, 9=max)
                "home_consumption",  # current home electricity usage (0-9)
                "battery_level",     # battery charge level (0-9: 0=empty, 9=full)
                "time_of_day",       # 0=morning, 1=afternoon, 2=evening, 3=night
            ],
            "state_bins": [10, 10, 10, 4],
            "n_actions": 3,
            "action_meanings": [
                "Use Solar Directly",   # power home with solar right now
                "Store in Battery",     # save solar for later use
                "Buy from Grid",        # supplement with grid electricity
            ],
            "objectives": ["Maximize solar usage", "Minimize grid dependency"],
        },

        # ------------------------------------------------------------------
        # TASK 2: Battery Management
        # Decides WHEN to charge or discharge the battery
        # ------------------------------------------------------------------
        "battery_management": {
            "description": "Battery storage charge/discharge optimization",
            "state_vars": [
                "battery_level",        # current battery charge (0-9: 0=empty, 9=full)
                "solar_output",         # current solar generation (0-9)
                "grid_price",           # electricity price (0=cheap, 1=normal, 2=expensive)
                "home_consumption",     # current home usage (0-9)
            ],
            "state_bins": [10, 10, 3, 10],
            "n_actions": 3,
            "action_meanings": [
                "Charge Battery",       # store energy (from solar or cheap grid)
                "Discharge Battery",    # use stored energy to power home
                "Keep Battery Idle",    # neither charge nor discharge
            ],
            "objectives": ["Maximize battery efficiency", "Minimize electricity cost"],
        },

        # ------------------------------------------------------------------
        # TASK 3: Grid Interaction
        # Decides WHEN to buy from or sell to the electricity grid
        # ------------------------------------------------------------------
        "grid_interaction": {
            "description": "Grid energy buying and selling optimization",
            "state_vars": [
                "grid_price",       # electricity price (0=cheap, 1=normal, 2=expensive)
                "solar_surplus",    # extra solar beyond home needs (0-9)
                "battery_level",    # current battery charge (0-9)
                "home_consumption", # current home usage (0-9)
            ],
            "state_bins": [3, 10, 10, 10],
            "n_actions": 3,
            "action_meanings": [
                "Buy from Grid",        # purchase electricity from grid
                "Sell to Grid",         # sell surplus solar to grid
                "Stay Self-Sufficient", # use own solar/battery only
            ],
            "objectives": ["Minimize electricity cost", "Maximize earnings from surplus"],
        },
    }

    def __init__(self, task: str = "solar_scheduling", render_mode=None):
        super().__init__()
        assert task in self.TASK_CONFIG, \
            f"Unknown task '{task}'. Choose from: {list(self.TASK_CONFIG.keys())}"

        self.task        = task
        self.render_mode = render_mode
        self.cfg         = self.TASK_CONFIG[task]

        state_size = int(np.prod(self.cfg["state_bins"]))
        self.observation_space = spaces.Discrete(state_size)
        self.action_space      = spaces.Discrete(self.cfg["n_actions"])

        self._raw_state  = None
        self._state      = None
        self._step_count = 0
        self.max_steps   = 100

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
        bins   = self.cfg["state_bins"]
        values = []
        for b in reversed(bins):
            values.append(state_idx % b)
            state_idx //= b
        return list(reversed(values))

    def _sample_raw_state(self) -> list:
        return [np.random.randint(0, b) for b in self.cfg["state_bins"]]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._raw_state  = self._sample_raw_state()
        self._state      = self._encode_state(self._raw_state)
        self._step_count = 0
        return self._state, {}

    def step(self, action: int):
        assert self.action_space.contains(action)
        reward          = self._compute_reward(action)
        self._raw_state = self._next_raw_state(action)
        self._state     = self._encode_state(self._raw_state)
        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        return self._state, reward, terminated, False, {}

    # ------------------------------------------------------------------
    # Reward logic
    # ------------------------------------------------------------------
    def _compute_reward(self, action: int) -> float:

        if self.task == "solar_scheduling":
            solar, consumption, battery, time = self._raw_state

            if action == 0:   # Use Solar Directly
                # Good when solar output meets or exceeds consumption
                match = min(solar, consumption)
                reward = match * 2.0
                if solar == 0:
                    reward -= 4.0   # penalty: no solar to use!
                if solar > consumption:
                    reward += (solar - consumption) * 0.5  # bonus: surplus available

            elif action == 1: # Store in Battery
                # Good when solar is high and battery is not full
                if solar > 0 and battery < 9:
                    reward = solar * 1.5 - (battery / 9) * 2.0
                elif battery >= 9:
                    reward = -3.0   # battery already full
                else:
                    reward = -2.0   # no solar to store

            else:             # Buy from Grid
                # Necessary when solar is low and battery is empty
                if solar == 0 and battery == 0:
                    reward = 2.0    # reasonable choice
                elif solar > 0:
                    reward = -3.0   # wasteful — solar is available!
                else:
                    reward = 0.5

            return float(np.clip(reward, -10, 10))

        elif self.task == "battery_management":
            battery, solar, price, consumption = self._raw_state

            if action == 0:   # Charge Battery
                # Best when solar is available or grid is cheap and battery not full
                if solar > 3 and battery < 8:
                    reward = solar * 1.5 + (1 - price) * 2.0
                elif price == 0 and battery < 8:   # cheap grid, good to charge
                    reward = 2.0
                elif battery >= 8:
                    reward = -3.0   # battery already full
                else:
                    reward = 0.5

            elif action == 1: # Discharge Battery
                # Best when grid is expensive and battery has charge
                if battery > 2 and price == 2:
                    reward = battery * 1.5 + consumption * 0.5
                elif battery > 2 and consumption > 5:
                    reward = battery * 1.0
                elif battery <= 2:
                    reward = -4.0   # battery nearly empty!
                else:
                    reward = 0.0

            else:             # Keep Idle
                # Reasonable when price is normal and solar covers consumption
                if solar >= consumption:
                    reward = 2.0    # solar covers everything, no need for battery
                elif price == 1:
                    reward = 1.0    # normal price, idle is fine
                else:
                    reward = -1.0

            return float(np.clip(reward, -10, 10))

        elif self.task == "grid_interaction":
            price, surplus, battery, consumption = self._raw_state

            if action == 0:   # Buy from Grid
                # Reasonable when battery empty and no solar surplus
                if battery == 0 and surplus == 0:
                    reward = 3.0 - price * 2.0   # better when price is low
                elif surplus > 0 or battery > 3:
                    reward = -3.0   # wasteful — own resources available!
                else:
                    reward = 1.0 - price * 1.5

            elif action == 1: # Sell to Grid
                # Best when surplus is high and price is high
                if surplus > 3:
                    reward = surplus * 1.5 + price * 2.0
                elif surplus == 0:
                    reward = -4.0   # nothing to sell!
                else:
                    reward = surplus * 0.8

            else:             # Stay Self-Sufficient
                # Best when solar + battery can cover consumption
                own_power = surplus + battery
                if own_power >= consumption:
                    reward = 4.0 + price * 1.0   # great — fully independent!
                elif own_power > 0:
                    reward = own_power * 0.5
                else:
                    reward = -2.0   # no resources to be self-sufficient

            return float(np.clip(reward, -10, 10))

        return 0.0

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _next_raw_state(self, action: int) -> list:
        raw  = self._raw_state.copy()
        bins = self.cfg["state_bins"]

        if self.task == "solar_scheduling":
            solar, consumption, battery, time = raw

            if action == 0:   # Use Solar Directly
                solar       = max(0, solar - np.random.randint(1, 3))
                consumption = max(0, consumption - np.random.randint(1, 3))
            elif action == 1: # Store in Battery
                battery = min(bins[2]-1, battery + min(solar, np.random.randint(1, 3)))
                solar   = max(0, solar - np.random.randint(1, 3))
            else:             # Buy from Grid
                consumption = max(0, consumption - np.random.randint(1, 3))

            # Natural changes over time
            time  = (time + np.random.randint(0, 2)) % 4
            # Solar depends on time of day
            if time == 1:   # afternoon — peak solar
                solar = min(bins[0]-1, solar + np.random.randint(1, 4))
            elif time == 0: # morning — moderate solar
                solar = min(bins[0]-1, solar + np.random.randint(0, 3))
            else:           # evening/night — low/no solar
                solar = max(0, solar - np.random.randint(0, 3))
            consumption = min(bins[1]-1, consumption + np.random.randint(0, 3))
            raw = [solar, consumption, battery, time]

        elif self.task == "battery_management":
            battery, solar, price, consumption = raw

            if action == 0:   # Charge Battery
                battery = min(bins[0]-1, battery + np.random.randint(1, 3))
                solar   = max(0, solar - np.random.randint(0, 2))
            elif action == 1: # Discharge Battery
                battery     = max(0, battery - np.random.randint(1, 3))
                consumption = max(0, consumption - np.random.randint(1, 3))
            # else: Idle — no battery change

            # Price fluctuates randomly
            price = int(np.clip(price + np.random.randint(-1, 2), 0, 2))
            solar = min(bins[1]-1, max(0, solar + np.random.randint(-2, 3)))
            consumption = min(bins[3]-1, max(0, consumption + np.random.randint(-1, 2)))
            raw = [battery, solar, price, consumption]

        elif self.task == "grid_interaction":
            price, surplus, battery, consumption = raw

            if action == 0:   # Buy from Grid
                consumption = max(0, consumption - np.random.randint(1, 3))
            elif action == 1: # Sell to Grid
                surplus = max(0, surplus - np.random.randint(1, 3))
            else:             # Self-Sufficient
                battery  = max(0, battery - np.random.randint(0, 2))
                surplus  = max(0, surplus - np.random.randint(0, 2))

            # Natural state changes
            price   = int(np.clip(price + np.random.randint(-1, 2), 0, 2))
            surplus = min(bins[1]-1, max(0, surplus + np.random.randint(0, 3)))
            battery = min(bins[2]-1, max(0, battery + np.random.randint(-1, 2)))
            consumption = min(bins[3]-1, max(0, consumption + np.random.randint(-1, 2)))
            raw = [price, surplus, battery, consumption]

        return [int(np.clip(raw[i], 0, self.cfg["state_bins"][i]-1))
                for i in range(len(raw))]

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        print(f"\n[Energy-{self.task.upper()}] Step {self._step_count}")
        for var, val in zip(self.cfg["state_vars"], self._raw_state):
            print(f"  {var}: {val}")

    def get_state_info(self):
        return dict(zip(self.cfg["state_vars"], self._raw_state))
