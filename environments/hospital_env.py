"""
UMORDA — Hospital Environment
Gymnasium-compatible multi-task hospital resource management environment.
Supports multi-step episodes so the agent learns across time, not just one-shot.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class HospitalEnv(gym.Env):
    """
    A Gymnasium-style environment for hospital resource management.

    Three tasks share this class:
      - bed_allocation   : decide whether to Admit / Transfer / Reject incoming patients
      - er_queue         : triage between Emergency and Normal queue each timestep
      - staff_allocation : adjust doctor count relative to patient load

    Each episode runs for MAX_STEPS timesteps (not done after 1 action).
    This forces the agent to learn sequential, consistent decision-making.
    """

    metadata = {"render_modes": ["human"]}

    # ── Per-task episode length ──────────────────────────────────────────────
    MAX_STEPS = {
        "bed_allocation":   20,
        "er_queue":         20,
        "staff_allocation": 20,
    }

    def __init__(self, task="bed_allocation", render_mode=None):
        super().__init__()
        assert task in self.MAX_STEPS, f"Unknown task: {task}"
        self.task        = task
        self.render_mode = render_mode
        self.state       = {}
        self.step_count  = 0
        self._setup_spaces()

    # ── Action / Observation spaces ──────────────────────────────────────────
    def _setup_spaces(self):
        if self.task == "bed_allocation":
            self.actions    = ["Admit", "Transfer", "Reject"]
            # obs: [free_beds (0-20), waiting_patients (0-30)]
            self.observation_space = spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([20, 30], dtype=np.float32),
            )

        elif self.task == "er_queue":
            self.actions    = ["Serve Emergency", "Serve Normal"]
            # obs: [emergency_queue (0-10), normal_queue (0-20)]
            self.observation_space = spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([10, 20], dtype=np.float32),
            )

        elif self.task == "staff_allocation":
            self.actions    = ["Assign More Staff", "Keep Current", "Reduce Staff"]
            # obs: [available_doctors (1-15), patient_load (0-50)]
            self.observation_space = spaces.Box(
                low=np.array([1, 0], dtype=np.float32),
                high=np.array([15, 50], dtype=np.float32),
            )

        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        if self.task == "bed_allocation":
            self.state = {
                "free_beds":        random.randint(0, 20),
                "waiting_patients": random.randint(1, 30),   # always ≥1 so there's always a decision
            }

        elif self.task == "er_queue":
            # Ensure at least one queue is non-empty
            self.state = {
                "emergency_queue": random.randint(0, 10),
                "normal_queue":    random.randint(0, 20),
            }
            if self.state["emergency_queue"] == 0 and self.state["normal_queue"] == 0:
                self.state["normal_queue"] = random.randint(1, 5)

        elif self.task == "staff_allocation":
            self.state = {
                "available_doctors": random.randint(1, 15),
                "patient_load":      random.randint(0, 50),
            }

        return self._obs(), {}

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action_index):
        assert 0 <= action_index < self.n_actions
        action  = self.actions[action_index]
        reward  = 0.0
        info    = {}

        # ── Bed Allocation ───────────────────────────────────────────────────
        if self.task == "bed_allocation":
            free_beds        = self.state["free_beds"]
            waiting_patients = self.state["waiting_patients"]

            if action == "Admit":
                if free_beds > 10:
                    reward = +10.0
                    info["result"] = "Patient admitted — ample beds"
                elif free_beds > 5:
                    reward = +7.0
                    info["result"] = "Patient admitted — moderate capacity"
                elif free_beds > 0:
                    reward = +3.0
                    info["result"] = "Patient admitted — beds running low"
                else:
                    reward = -15.0
                    info["result"] = "Cannot admit — no free beds (penalty)"

                if free_beds > 0:
                    self.state["free_beds"]        = max(0, free_beds - 1)
                    self.state["waiting_patients"] = max(0, waiting_patients - 1)

            elif action == "Transfer":
             if free_beds == 0:
              reward = +12.0   # was +10
             elif free_beds <= 5:
              reward = +8.0    # was +6
             else:
              reward = -6.0    # unchanged

            elif action == "Reject":
              reward = -20.0   # was -12, raise it so Transfer always wins
              info["result"] = "Patient rejected — unacceptable"
              # Stochastic arrivals each step — environment keeps moving
              new_arrivals = random.randint(0, 3)
              self.state["waiting_patients"] = min(30, self.state["waiting_patients"] + new_arrivals)
              self.state["free_beds"]        = min(20, self.state["free_beds"] + random.randint(0, 1))  # occasional discharge

        # ── ER Queue ─────────────────────────────────────────────────────────
        elif self.task == "er_queue":
            emergency_queue = self.state["emergency_queue"]
            normal_queue    = self.state["normal_queue"]

            if action == "Serve Emergency":
                if emergency_queue > 0:
                    reward = +10.0
                    self.state["emergency_queue"] -= 1
                    info["result"] = "Emergency patient served"
                else:
                    reward = -5.0
                    info["result"] = "No emergency patients — wasted action"

            elif action == "Serve Normal":
                if normal_queue > 0:
                    if emergency_queue > 0:
                        # Strong penalty — skipping critical patients
                        penalty = -5.0 - emergency_queue * 1.5   # scales with backlog
                        reward  = penalty
                        info["result"] = f"Skipped emergency queue (penalty={penalty:.1f})"
                    else:
                        reward = +8.0
                        info["result"] = "Normal patient served — no emergency backlog"
                    self.state["normal_queue"] -= 1
                else:
                    reward = -5.0
                    info["result"] = "No normal patients — wasted action"

            # Stochastic arrivals
            self.state["emergency_queue"] = min(10, self.state["emergency_queue"] + random.randint(0, 2))
            self.state["normal_queue"]    = min(20, self.state["normal_queue"]    + random.randint(0, 3))

        # ── Staff Allocation ──────────────────────────────────────────────────
        elif self.task == "staff_allocation":
            available_doctors = self.state["available_doctors"]
            patient_load      = self.state["patient_load"]

            if action == "Assign More Staff":
                if patient_load > 30:
                    reward = +10.0
                    self.state["available_doctors"] = min(15, available_doctors + 1)
                    info["result"] = "Staff added — high load justified"
                elif patient_load > 15:
                    reward = +4.0
                    self.state["available_doctors"] = min(15, available_doctors + 1)
                    info["result"] = "Staff added — moderate load, acceptable"
                else:
                    reward = -8.0
                    info["result"] = "Overstaffed — low patient load"

            elif action == "Keep Current":
                if 10 <= patient_load <= 30:
                    reward = +8.0
                    info["result"] = "Maintained staffing — load is balanced"
                elif patient_load < 10:
                    reward = -4.0
                    info["result"] = "Should reduce staff — load too low"
                else:
                    reward = -4.0
                    info["result"] = "Should add staff — load too high"

            elif action == "Reduce Staff":
                if patient_load < 10:
                    reward = +10.0
                    self.state["available_doctors"] = max(1, available_doctors - 1)
                    info["result"] = "Staff reduced — low load, cost saved"
                elif patient_load <= 20:
                    reward = -4.0
                    info["result"] = "Risky reduction — moderate load"
                else:
                    reward = -15.0
                    self.state["available_doctors"] = max(1, available_doctors - 1)
                    info["result"] = "DANGEROUS — reduced staff under high load"

            # Stochastic load shift each step
            load_delta = random.randint(-5, 8)
            self.state["patient_load"] = int(np.clip(patient_load + load_delta, 0, 50))

        # ── Episode bookkeeping ───────────────────────────────────────────────
        self.step_count += 1
        max_steps   = self.MAX_STEPS[self.task]
        terminated  = False
        truncated   = self.step_count >= max_steps
        info["step"] = self.step_count

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, info

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _obs(self):
        if self.task == "bed_allocation":
            return np.array([self.state["free_beds"], self.state["waiting_patients"]], dtype=np.float32)
        elif self.task == "er_queue":
            return np.array([self.state["emergency_queue"], self.state["normal_queue"]], dtype=np.float32)
        elif self.task == "staff_allocation":
            return np.array([self.state["available_doctors"], self.state["patient_load"]], dtype=np.float32)

    def get_state_dict(self):
        return dict(self.state)

    def render(self):
        print(f"  [Step {self.step_count:02d}] {self.task} | State: {self.state}")

    def get_info(self):
        return {
            "task":       self.task,
            "actions":    self.actions,
            "n_actions":  self.n_actions,
            "max_steps":  self.MAX_STEPS[self.task],
        }