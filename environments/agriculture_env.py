"""
UMORDA — Agriculture Environment (Gymnasium-compatible)
Multi-task growing-season simulation. Each episode is a "season" of up to
SHIFT_LENGTH steps (fewer if the agent reaches a real terminal decision,
like "Plant Now"). State carries over between steps so a discount factor
(GAMMA) has real meaning.

Follows the same structural pattern as HospitalEnv and FinanceEnv:
  - gym.Env subclass
  - observation_space / action_space declared via gymnasium.spaces
  - reset() returns (obs, info)
  - step() returns (obs, reward, terminated, truncated, info)
  - reward is decomposed into r_performance / r_cost / r_fairness, each
    logged in info for later analysis (same convention as FinanceEnv)

Unlike HospitalEnv, `soil_preparation` uses a genuine `terminated` signal
(the agent can choose "Plant Now" and end the episode early) alongside
`truncated` (running out of the planting window without ever planting).
This gives the agent a real terminal state to learn from, not just a
time-out.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class AgricultureEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    SHIFT_LENGTH = 40  # steps per episode ("season"), used by irrigation/pest_control

    def __init__(self, task="soil_preparation", render_mode=None):
        super().__init__()
        self.task = task
        self.render_mode = render_mode
        self.state = None
        self.t = 0

        if self.task == "soil_preparation":
            # Preparing soil for a unique/exotic fruit crop (targets below are
            # roughly calibrated to dragon fruit / pitaya: slightly acidic to
            # neutral pH, moderate-to-high organic matter, and good drainage
            # since the plant is a cactus-family epiphyte that rots in wet feet).
            self.state_vars = ["soil_ph", "organic_matter", "drainage_quality", "days_remaining"]
            self.actions    = ["Add Compost", "Adjust pH", "Improve Drainage", "Plant Now"]
            self.n_actions  = 4
            self.PH_TARGET_LOW, self.PH_TARGET_HIGH = 5.5, 7.0
            self.OM_TARGET       = 60   # organic matter %, target >= this
            self.DRAINAGE_TARGET = 60   # drainage quality %, target >= this
            self.PLANTING_WINDOW = 25   # days before the window closes
            self.observation_space = spaces.Box(
                low=np.array([3.0, 0, 0, 0], dtype=np.float32),
                high=np.array([9.0, 100, 100, self.PLANTING_WINDOW], dtype=np.float32),
            )

        elif self.task == "irrigation":
            self.state_vars = ["water_reservoir", "crop_stress", "rainfall_trend", "days_remaining"]
            self.actions    = ["Irrigate Heavy", "Irrigate Light", "Skip Irrigation"]
            self.n_actions  = 3
            # water_reservoir/crop_stress: 0-100, rainfall_trend: -2 (drought) to +2 (heavy rain)
            self.observation_space = spaces.Box(
                low=np.array([0, 0, -2, 0], dtype=np.float32),
                high=np.array([100, 100, 2, self.SHIFT_LENGTH], dtype=np.float32),
            )

        elif self.task == "pest_control":
            self.state_vars = ["total_resource", "resource_used", "urgent_outbreaks", "plots_remaining"]
            self.actions    = ["Full Treatment", "Partial Treatment", "Defer"]
            self.n_actions  = 3
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0], dtype=np.float32),
                high=np.array([1000, 1000, 10, 10], dtype=np.float32),
            )

        else:
            raise ValueError(f"Unknown task: {task}")

        self.action_space = spaces.Discrete(self.n_actions)

    # ── Reset: start of a new season ─────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        if self.task == "soil_preparation":
            self.state = {
                "soil_ph":          round(random.uniform(4.0, 8.0), 2),
                "organic_matter":   float(random.randint(10, 40)),
                "drainage_quality": float(random.randint(10, 40)),
                "days_remaining":   float(self.PLANTING_WINDOW),
            }

        elif self.task == "irrigation":
            self.state = {
                "water_reservoir": float(random.randint(40, 100)),
                "crop_stress":     float(random.randint(0, 30)),
                "rainfall_trend":  float(random.choice([-2, -1, 0, 1, 2])),
                "days_remaining":  float(self.SHIFT_LENGTH),
            }

        elif self.task == "pest_control":
            self.state = {
                "total_resource":   float(random.randint(500, 1000)),
                "resource_used":    0.0,
                "urgent_outbreaks": float(random.randint(0, 5)),
                "plots_remaining":  float(random.randint(4, 10)),
            }

        return self._obs(), {}

    # ── Step: one decision within the season ─────────────────────────────────
    def step(self, action_index):
        action = self.actions[action_index]
        terminated = False

        if self.task == "soil_preparation":
            reward, terminated, info = self._step_soil_preparation(action)
        elif self.task == "irrigation":
            reward, info = self._step_irrigation(action)
        elif self.task == "pest_control":
            reward, info = self._step_pest_control(action)

        self.t += 1

        if self.task == "soil_preparation":
            self.state["days_remaining"] = max(0, self.PLANTING_WINDOW - self.t)
            truncated = (self.t >= self.PLANTING_WINDOW) and not terminated
            if truncated:
                reward -= 20.0
                info["missed_window"] = True
                info["result"] = info.get("result", "") + " | Planting window closed -- never planted!"
        elif self.task == "irrigation":
            self.state["days_remaining"] = max(0, self.SHIFT_LENGTH - self.t)
            truncated = self.t >= self.SHIFT_LENGTH
        else:
            truncated = self.t >= self.SHIFT_LENGTH

        info["step"] = self.t

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════
    # Task 1 — Soil Preparation (unique fruit crop)
    # ══════════════════════════════════════════════════════════════════════
    def _step_soil_preparation(self, action):
        ph       = self.state["soil_ph"]
        om       = self.state["organic_matter"]
        drainage = self.state["drainage_quality"]
        terminated = False
        info = {}

        if action == "Add Compost":
            gain = random.uniform(8, 15)
            self.state["organic_matter"] = min(100, om + gain)
            # compost is mildly acidic -- nudges pH down slightly as a side effect
            self.state["soil_ph"] = max(3.0, ph - random.uniform(0, 0.15))
            r_perf = 4.0 if om < self.OM_TARGET else 1.0
            r_cost = -2.0   # labor + material cost
            r_fair = 0.0
            info["result"] = f"Added compost -- organic matter now {self.state['organic_matter']:.1f}%"

        elif action == "Adjust pH":
            # env nudges pH toward the target range (lime raises, sulfur lowers,
            # but the agent doesn't need to pick a direction -- keeps the
            # action space simple, same design choice as HospitalEnv's staffing task)
            if ph < self.PH_TARGET_LOW:
                self.state["soil_ph"] = min(9.0, ph + random.uniform(0.3, 0.6))
            elif ph > self.PH_TARGET_HIGH:
                self.state["soil_ph"] = max(3.0, ph - random.uniform(0.3, 0.6))
            else:
                self.state["soil_ph"] = ph + random.uniform(-0.1, 0.1)
            in_range_now = self.PH_TARGET_LOW <= self.state["soil_ph"] <= self.PH_TARGET_HIGH
            r_perf = 5.0 if in_range_now else 1.0
            r_cost = -2.0
            r_fair = 0.0
            info["result"] = f"Adjusted pH -- now {self.state['soil_ph']:.2f}"

        elif action == "Improve Drainage":
            gain = random.uniform(8, 15)
            self.state["drainage_quality"] = min(100, drainage + gain)
            r_perf = 4.0 if drainage < self.DRAINAGE_TARGET else 1.0
            r_cost = -3.0   # sand/grit amendment, pricier than compost
            r_fair = 0.0
            info["result"] = f"Improved drainage -- now {self.state['drainage_quality']:.1f}%"

        elif action == "Plant Now":
            ph_ok    = self.PH_TARGET_LOW <= self.state["soil_ph"] <= self.PH_TARGET_HIGH
            om_ok    = self.state["organic_matter"] >= self.OM_TARGET
            drain_ok = self.state["drainage_quality"] >= self.DRAINAGE_TARGET
            checks_passed = sum([ph_ok, om_ok, drain_ok])

            if checks_passed == 3:
                r_perf = 30.0
                info["result"] = "Planted -- soil fully ready, excellent conditions for the crop"
            elif checks_passed == 2:
                r_perf = 8.0
                info["result"] = "Planted -- soil mostly ready, crop should establish reasonably"
            elif checks_passed == 1:
                r_perf = -10.0
                info["result"] = "Planted -- soil poorly prepared, high risk of crop failure"
            else:
                r_perf = -25.0
                info["result"] = "Planted -- soil unready on all fronts, crop likely to fail"

            r_cost, r_fair = 0.0, 0.0
            terminated = True
            info["ph_ok"], info["om_ok"], info["drainage_ok"] = ph_ok, om_ok, drain_ok

        reward = r_perf + r_cost + r_fair
        info["r_performance"] = r_perf
        info["r_cost"]        = r_cost
        info["r_fairness"]    = r_fair
        return reward, terminated, info

    # ══════════════════════════════════════════════════════════════════════
    # Task 2 — Irrigation Management
    # ══════════════════════════════════════════════════════════════════════
    def _step_irrigation(self, action):
        reservoir = self.state["water_reservoir"]
        stress    = self.state["crop_stress"]
        trend     = self.state["rainfall_trend"]
        info = {}

        if action == "Irrigate Heavy":
            use = 15
            if reservoir >= use:
                self.state["water_reservoir"] -= use
                self.state["crop_stress"] = max(0, stress - 20)
                r_perf = 8.0 if stress > 40 else 2.0   # more valuable when crop was actually stressed
                r_cost = -4.0
                info["result"] = "Heavy irrigation -- crop stress relieved"
            else:
                r_perf, r_cost = -5.0, 0.0
                info["result"] = "Not enough water in reservoir for heavy irrigation"

        elif action == "Irrigate Light":
            use = 6
            if reservoir >= use:
                self.state["water_reservoir"] -= use
                self.state["crop_stress"] = max(0, stress - 8)
                r_perf = 4.0 if stress > 20 else 1.0
                r_cost = -1.5
                info["result"] = "Light irrigation applied"
            else:
                r_perf, r_cost = -3.0, 0.0
                info["result"] = "Not enough water for even light irrigation"

        elif action == "Skip Irrigation":
            r_cost = +2.0   # water conserved
            if trend <= -1:   # dry spell -- skipping is risky
                self.state["crop_stress"] = min(100, stress + 15)
                r_perf = -6.0
                info["result"] = "Skipped irrigation during dry spell -- crop stress rising"
            else:
                self.state["crop_stress"] = min(100, stress + 3)
                r_perf = 1.0
                info["result"] = "Skipped irrigation -- conditions mild, minor stress increase"

        r_fair = 0.0  # single field, not deeply applicable -- kept neutral (same as FinanceEnv trading)

        # World dynamics: rainfall trend drifts, reservoir refills with rain
        new_trend = int(np.clip(trend + random.choice([-1, 0, 0, 0, 1]), -2, 2))
        rainfall_gain = {-2: 0, -1: 2, 0: 6, 1: 12, 2: 20}[new_trend]
        self.state["rainfall_trend"]  = new_trend
        self.state["water_reservoir"] = min(100, self.state["water_reservoir"] + rainfall_gain)

        # Hard penalty if crop stress maxes out (crop close to dying)
        if self.state["crop_stress"] >= 90:
            r_perf -= 10.0
            info["critical_stress"] = True

        reward = r_perf + r_cost + r_fair
        info["r_performance"] = r_perf
        info["r_cost"]        = r_cost
        info["r_fairness"]    = r_fair
        info["rainfall_gain"] = rainfall_gain
        return reward, info

    # ══════════════════════════════════════════════════════════════════════
    # Task 3 — Pest Control / Treatment Allocation
    # ══════════════════════════════════════════════════════════════════════
    def _step_pest_control(self, action):
        total  = self.state["total_resource"]
        used   = self.state["resource_used"]
        urgent = self.state["urgent_outbreaks"]
        plots  = self.state["plots_remaining"]
        remaining = total - used
        info = {}

        request = float(random.randint(40, 150))  # cost to treat this plot's outbreak
        info["request_size"] = request

        if action == "Full Treatment":
            if remaining >= request:
                self.state["resource_used"] += request
                if urgent > 0:
                    r_perf, r_fair, r_cost = 15.0, 5.0, -3.0
                    self.state["urgent_outbreaks"] = max(0, urgent - 1)
                    info["result"] = f"Fully treated urgent outbreak (${request:.0f})"
                else:
                    r_perf, r_fair, r_cost = 6.0, 2.0, -1.0
                    info["result"] = f"Fully treated plot (${request:.0f})"
            else:
                reward = -10.0
                info["result"] = f"Over budget -- cannot fully treat (${request:.0f} needed, ${remaining:.0f} left)"
                info["r_performance"], info["r_cost"], info["r_fairness"] = 0.0, 0.0, 0.0
                return self._finalise_pest_control(reward, info)

        elif action == "Partial Treatment":
            ratio   = random.uniform(0.4, 0.7)
            partial = request * ratio
            if remaining >= partial:
                self.state["resource_used"] += partial
                if urgent > 0:
                    r_perf, r_fair, r_cost = 5.0, -2.0, 3.0
                    info["result"] = f"Partially treated urgent outbreak (${partial:.0f} of ${request:.0f})"
                else:
                    r_perf, r_fair, r_cost = 4.0, 1.0, 5.0
                    info["result"] = f"Smart partial treatment (${partial:.0f} of ${request:.0f})"
            else:
                reward = -6.0
                info["result"] = "Even partial treatment exceeds remaining resources"
                info["r_performance"], info["r_cost"], info["r_fairness"] = 0.0, 0.0, 0.0
                return self._finalise_pest_control(reward, info)

        elif action == "Defer":
            if urgent > 0:
                r_perf, r_fair, r_cost = -15.0, -8.0, 5.0
                info["result"] = "Deferred URGENT outbreak -- crop damage risk"
            elif plots <= 1:
                r_perf, r_fair, r_cost = -8.0, -3.0, 3.0
                info["result"] = "Deferred last plot -- poor planning"
            else:
                r_perf, r_fair, r_cost = -2.0, 0.0, 6.0
                info["result"] = "Deferred non-urgent outbreak -- resources conserved"

        reward = r_perf + r_fair + r_cost

        if self.state["resource_used"] > total * 1.05:
            reward -= 20.0
            info["over_budget_penalty"] = True

        if plots <= 1 and remaining > 0 and urgent == 0:
            reward += 8.0
            info["clean_finish_bonus"] = remaining

        info["r_performance"] = r_perf
        info["r_cost"]        = r_cost
        info["r_fairness"]    = r_fair
        return self._finalise_pest_control(reward, info)

    def _finalise_pest_control(self, reward, info):
        """Apply world dynamics for pest_control task."""
        urgent = self.state["urgent_outbreaks"]
        plots  = self.state["plots_remaining"]

        new_urgent = random.choices([0, 1, 2], weights=[60, 30, 10])[0]
        self.state["urgent_outbreaks"] = min(10, urgent + new_urgent)
        self.state["plots_remaining"]  = max(0, plots - 1)

        info["remaining_resource"] = round(self.state["total_resource"] - self.state["resource_used"], 1)
        info["urgent_remaining"]   = self.state["urgent_outbreaks"]
        info["plots_left"]        = self.state["plots_remaining"]
        return reward, info

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _obs(self):
        if self.task == "soil_preparation":
            return np.array([
                self.state["soil_ph"],
                self.state["organic_matter"],
                self.state["drainage_quality"],
                self.state["days_remaining"],
            ], dtype=np.float32)
        elif self.task == "irrigation":
            return np.array([
                self.state["water_reservoir"],
                self.state["crop_stress"],
                self.state["rainfall_trend"],
                self.state["days_remaining"],
            ], dtype=np.float32)
        elif self.task == "pest_control":
            return np.array([
                self.state["total_resource"],
                self.state["resource_used"],
                self.state["urgent_outbreaks"],
                self.state["plots_remaining"],
            ], dtype=np.float32)

    def render(self):
        print(f"  [t={self.t:02d}] {self.task} | {self.state}")

    def get_info(self):
        return {
            "task":        self.task,
            "state_vars":  self.state_vars,
            "actions":     self.actions,
            "shift_length": self.SHIFT_LENGTH,
        }
