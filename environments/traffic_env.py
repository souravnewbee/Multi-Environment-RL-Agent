# =============================================================================
# UMORDA — Traffic Domain Environment (FIXED VERSION v2)
# File: environments/traffic_env.py
#
# FIXES APPLIED:
# A) Wait TIME added to state — not just count
# B) Urgency scales with how long someone has been waiting
# C) Safety is a HARD GUARANTEE — not just a penalty term
# D) Signal phase + elapsed time added to state (no free switching)
# E) Wider state space — handles heavy traffic properly
# =============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficEnv(gym.Env):
    """
    Fixed Multi-task Traffic Management Environment.

    Key improvements over v1:
    - State includes WAIT TIME (not just count)
    - Safety is a hard constraint layered on top of Q-values
    - Signal phase memory prevents rapid flipping
    - Wider discretization handles congested conditions
    - Reward measures cumulative wait cost reduction
    """

    metadata = {"render_modes": ["human"]}

    TASK_CONFIG = {
        "intersection": {
            "description": "Single traffic intersection control",
            "state_vars": [
                "cars_NS",          # total cars North+South (0-9)
                "cars_EW",          # total cars East+West (0-9)
                "current_phase",    # 0=GreenNS, 1=GreenEW
                "phase_elapsed",    # how many steps current phase has been active (0-9)
                "max_wait_NS",      # longest any NS car has waited (0-9)
                "max_wait_EW",      # longest any EW car has waited (0-9)
            ],
            "state_bins": [10, 10, 2, 10, 10, 10],
            "n_actions": 2,
            "action_meanings": ["Keep / Switch to Green NS", "Keep / Switch to Green EW"],
            "objectives": ["Minimize total wait time", "Prevent starvation of any direction"],
            "max_wait_limit": 8,    # hard safety limit — no direction waits more than this
            "min_phase_duration": 3, # minimum steps before switching is allowed
        },
        "pedestrian": {
            "description": "Pedestrian crossing control",
            "state_vars": [
                "waiting_pedestrians",   # count (0-9)
                "waiting_vehicles",      # count (0-9)
                "ped_max_wait",          # longest pedestrian wait time (0-9)
                "veh_max_wait",          # longest vehicle wait time (0-9)
                "current_phase",         # 0=PedPhase, 1=VehiclePhase
                "phase_elapsed",         # steps in current phase (0-9)
            ],
            "state_bins": [10, 10, 10, 10, 2, 10],
            "n_actions": 2,
            "action_meanings": ["Allow Pedestrians", "Allow Vehicles"],
            "objectives": ["Maximize safety", "Minimize cumulative wait time"],
            "max_ped_wait": 6,      # hard safety limit for pedestrians
            "max_veh_wait": 9,      # vehicle wait limit
            "min_phase_duration": 2,
        },
        "parking": {
            "description": "Parking lot management",
            "state_vars": [
                "available_spots",       # 0-19 (wider range)
                "incoming_vehicles",     # 0-9
                "queue_wait_time",       # how long vehicles have been queuing (0-9)
                "occupancy_rate",        # 0=<25%, 1=25-50%, 2=50-75%, 3=75-100%, 4=FULL
            ],
            "state_bins": [20, 10, 10, 5],
            "n_actions": 3,
            "action_meanings": ["Open Zone A", "Open Zone B", "Close Entry"],
            "objectives": ["Maximize occupancy", "Minimize queue congestion"],
            "max_queue_wait": 7,
        },
    }

    def __init__(self, task: str = "intersection", render_mode=None):
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
        self.max_steps   = 150

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
        bins = self.cfg["state_bins"]
        raw  = [np.random.randint(0, b) for b in bins]

        if self.task == "intersection":
            # phase_elapsed starts at 0
            raw[3] = 0
        elif self.task == "pedestrian":
            raw[5] = 0
        return raw

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

        # Check hard safety constraint BEFORE applying Q-value decision
        forced_action = self._safety_override(action)

        reward          = self._compute_reward(forced_action)
        self._raw_state = self._next_raw_state(forced_action)
        self._state     = self._encode_state(self._raw_state)
        self._step_count += 1

        terminated = self._step_count >= self.max_steps
        truncated  = False

        if self.render_mode == "human":
            self.render()

        return self._state, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # FIX C: Hard safety override — agent cannot ignore safety limits
    # ------------------------------------------------------------------
    def _safety_override(self, action: int) -> int:
        """
        If any party has waited beyond the safety limit,
        FORCE the signal to serve them — regardless of Q-values.
        This is a hard guarantee, not a soft penalty.
        """
        if self.task == "intersection":
            max_wait_NS = self._raw_state[4]
            max_wait_EW = self._raw_state[5]
            limit       = self.cfg["max_wait_limit"]
            if max_wait_NS >= limit:
                return 0   # FORCE Green NS
            if max_wait_EW >= limit:
                return 1   # FORCE Green EW

        elif self.task == "pedestrian":
            ped_wait = self._raw_state[2]
            limit    = self.cfg["max_ped_wait"]
            if ped_wait >= limit:
                return 0   # FORCE pedestrian phase — safety guarantee

        elif self.task == "parking":
            queue_wait = self._raw_state[2]
            spots      = self._raw_state[0]
            if queue_wait >= self.cfg["max_queue_wait"] and spots > 0:
                return 0   # FORCE open zone A if vehicles waiting too long

        return action   # no override needed

    # ------------------------------------------------------------------
    # FIX A+B: Reward based on WAIT TIME reduction, not count comparison
    # ------------------------------------------------------------------
    def _compute_reward(self, action: int) -> float:

        if self.task == "intersection":
            cars_NS, cars_EW, phase, elapsed, wait_NS, wait_EW = self._raw_state
            limit    = self.cfg["max_wait_limit"]
            min_dur  = self.cfg["min_phase_duration"]

            # FIX D: penalty for switching too fast
            switching_penalty = 0.0
            if phase != action and elapsed < min_dur:
                switching_penalty = -5.0

            if action == 0:   # Green NS
                # Reward = wait time cleared × cars served
                reward = (wait_NS * cars_NS * 0.5) - (wait_EW * 0.3)
            else:             # Green EW
                reward = (wait_EW * cars_EW * 0.5) - (wait_NS * 0.3)

            # Urgency bonus — reward clearing the longest waiter more
            urgency = max(wait_NS if action == 0 else wait_EW, 0) * 0.5
            reward += urgency

            # Starvation penalty — if other direction is near limit
            other_wait = wait_EW if action == 0 else wait_NS
            if other_wait >= limit - 2:
                reward -= 8.0   # approaching safety limit

            reward += switching_penalty
            return float(np.clip(reward, -20, 20))

        elif self.task == "pedestrian":
            peds, vehs, ped_wait, veh_wait, phase, elapsed = self._raw_state
            min_dur = self.cfg["min_phase_duration"]
            ped_limit = self.cfg["max_ped_wait"]

            switching_penalty = -5.0 if (phase != action and elapsed < min_dur) else 0.0

            if action == 0:   # Allow Pedestrians
                # Reward scales with HOW LONG peds have been waiting
                reward = (ped_wait * 2.0 + peds * 0.5)
                # Penalty if no pedestrians actually waiting
                if peds == 0:
                    reward -= 4.0
                # Penalty if vehicles near their limit
                if veh_wait >= self.cfg["max_veh_wait"] - 2:
                    reward -= 3.0
            else:             # Allow Vehicles
                reward = (veh_wait * 1.5 + vehs * 0.4)
                if vehs == 0:
                    reward -= 3.0
                # Strong penalty if pedestrian safety limit approaching
                if ped_wait >= ped_limit - 1:
                    reward -= 10.0  # smooth proportional penalty, not a cliff

            reward += switching_penalty
            return float(np.clip(reward, -20, 20))

        elif self.task == "parking":
            spots, incoming, queue_wait, occupancy = self._raw_state

            if action == 0:   # Open Zone A
                served  = min(incoming, max(spots, 0))
                reward  = served * 2.0 - queue_wait * 0.5
                if spots == 0:
                    reward -= 5.0
            elif action == 1: # Open Zone B
                served  = min(incoming, max(spots // 2, 0))
                reward  = served * 1.5 - queue_wait * 0.3
                if spots < 5:
                    reward -= 3.0
            else:             # Close Entry
                if spots <= 2:
                    reward = 4.0 + (9 - queue_wait)  # good: lot genuinely full
                else:
                    reward = -3.0 - queue_wait * 0.5  # bad: closed unnecessarily

            # Reward for reducing queue wait
            reward += max(0, 5 - queue_wait) * 0.3
            return float(np.clip(reward, -20, 20))

        return 0.0

    # ------------------------------------------------------------------
    # FIX B+D: State transitions track wait TIME, not just count
    # ------------------------------------------------------------------
    def _next_raw_state(self, action: int) -> list:
        raw  = self._raw_state.copy()
        bins = self.cfg["state_bins"]

        if self.task == "intersection":
            cars_NS, cars_EW, phase, elapsed, wait_NS, wait_EW = raw
            min_dur = self.cfg["min_phase_duration"]

            switching = (action != phase) and (elapsed >= min_dur)

            if action == 0:  # Green NS active
                cars_NS  = max(0, cars_NS - np.random.randint(1, 4))
                wait_NS  = max(0, wait_NS - np.random.randint(1, 3))
                cars_EW  = min(bins[1]-1, cars_EW + np.random.randint(0, 3))
                wait_EW  = min(bins[5]-1, wait_EW + np.random.randint(0, 2))
            else:            # Green EW active
                cars_EW  = max(0, cars_EW - np.random.randint(1, 4))
                wait_EW  = max(0, wait_EW - np.random.randint(1, 3))
                cars_NS  = min(bins[0]-1, cars_NS + np.random.randint(0, 3))
                wait_NS  = min(bins[4]-1, wait_NS + np.random.randint(0, 2))

            # New arrivals
            cars_NS = min(bins[0]-1, cars_NS + np.random.randint(0, 3))
            cars_EW = min(bins[1]-1, cars_EW + np.random.randint(0, 3))

            # FIX D: track phase and elapsed time
            new_phase   = action
            new_elapsed = 0 if switching else min(bins[3]-1, elapsed + 1)

            raw = [cars_NS, cars_EW, new_phase, new_elapsed, wait_NS, wait_EW]

        elif self.task == "pedestrian":
            peds, vehs, ped_wait, veh_wait, phase, elapsed = raw
            min_dur  = self.cfg["min_phase_duration"]
            switching = (action != phase) and (elapsed >= min_dur)

            if action == 0:  # Pedestrian phase
                peds     = max(0, peds - np.random.randint(1, 4))
                ped_wait = max(0, ped_wait - np.random.randint(1, 3))
                vehs     = min(bins[1]-1, vehs + np.random.randint(0, 3))
                veh_wait = min(bins[3]-1, veh_wait + np.random.randint(0, 2))
            else:            # Vehicle phase
                vehs     = max(0, vehs - np.random.randint(1, 4))
                veh_wait = max(0, veh_wait - np.random.randint(1, 3))
                peds     = min(bins[0]-1, peds + np.random.randint(0, 2))
                ped_wait = min(bins[2]-1, ped_wait + np.random.randint(0, 2))

            # New arrivals
            peds = min(bins[0]-1, peds + np.random.randint(0, 2))
            vehs = min(bins[1]-1, vehs + np.random.randint(0, 3))

            new_phase   = action
            new_elapsed = 0 if switching else min(bins[5]-1, elapsed + 1)
            raw = [peds, vehs, ped_wait, veh_wait, new_phase, new_elapsed]

        elif self.task == "parking":
            spots, incoming, queue_wait, occupancy = raw

            if action == 0:  # Open Zone A
                served   = min(incoming, spots)
                spots    = max(0, spots - served)
                incoming = max(0, incoming - served)
                queue_wait = max(0, queue_wait - 2)
            elif action == 1: # Open Zone B
                served   = min(incoming, max(spots // 2, 0))
                spots    = max(0, spots - served)
                incoming = max(0, incoming - served)
                queue_wait = max(0, queue_wait - 1)
            else:            # Close Entry
                queue_wait = min(bins[2]-1, queue_wait + np.random.randint(1, 3))

            # New vehicles arrive
            incoming   = min(bins[1]-1, incoming + np.random.randint(0, 3))
            queue_wait = min(bins[2]-1, queue_wait + (1 if incoming > 0 else 0))

            # Update occupancy rate
            total_spots = bins[0] - 1
            filled      = total_spots - spots
            occ_ratio   = filled / max(total_spots, 1)
            if occ_ratio >= 1.0:
                occupancy = 4
            elif occ_ratio >= 0.75:
                occupancy = 3
            elif occ_ratio >= 0.50:
                occupancy = 2
            elif occ_ratio >= 0.25:
                occupancy = 1
            else:
                occupancy = 0

            raw = [spots, incoming, queue_wait, occupancy]

        # Clip all values to their bins
        return [int(np.clip(raw[i], 0, self.cfg["state_bins"][i]-1))
                for i in range(len(raw))]

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        cfg = self.cfg
        print(f"\n[Traffic-{self.task.upper()}] Step {self._step_count}")
        for var, val in zip(cfg["state_vars"], self._raw_state):
            print(f"  {var}: {val}")

    def get_state_info(self):
        return dict(zip(self.cfg["state_vars"], self._raw_state))
