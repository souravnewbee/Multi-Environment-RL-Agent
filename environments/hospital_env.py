import numpy as np
import random


class HospitalEnv:
    """
    Multi-step hospital simulation.

    Each episode is a full "shift" of SHIFT_LENGTH steps. State carries over
    between steps within a shift (beds stay occupied, queues persist, staff
    levels persist) so the agent's action now has real consequences later --
    which is the only way a discount factor (GAMMA) can mean anything.
    """

    SHIFT_LENGTH = 60  # steps per episode ("shift")

    def __init__(self, task="bed_allocation"):
        self.task = task
        self.state = None
        self.done = False
        self.t = 0

        if self.task == "bed_allocation":
            self.state_vars = ["free_beds", "waiting_patients"]
            self.actions = ["Admit", "Reject", "Transfer"]
            self.n_actions = 3
            self.total_beds = 20

        elif self.task == "er_queue":
            self.state_vars = ["emergency_queue", "normal_queue"]
            self.actions = ["Serve Emergency", "Serve Normal"]
            self.n_actions = 2

        elif self.task == "staff_allocation":
            self.state_vars = ["available_doctors", "patient_load"]
            self.actions = ["Assign More Staff", "Keep Current", "Reduce Staff"]
            self.n_actions = 3

        else:
            raise ValueError(f"Unknown task: {task}")

    # -- Reset: start of a new shift --------------------------
    def reset(self):
        self.t = 0
        self.done = False

        if self.task == "bed_allocation":
            self.state = {
                "free_beds": random.randint(0, self.total_beds),
                "waiting_patients": random.randint(0, 30),
            }
            occupied = self.total_beds - self.state["free_beds"]
            self._discharge_timers = [random.randint(1, 8) for _ in range(occupied)]

        elif self.task == "er_queue":
            self.state = {
                "emergency_queue": random.randint(0, 10),
                "normal_queue": random.randint(0, 20),
            }

        elif self.task == "staff_allocation":
            self.state = {
                "available_doctors": random.randint(1, 15),
                "patient_load": random.randint(0, 50),
            }

        return self.state

    # -- Step: one decision within the shift ------------------
    def step(self, action_index):
        action = self.actions[action_index]

        if self.task == "bed_allocation":
            reward, info = self._step_bed_allocation(action)
        elif self.task == "er_queue":
            reward, info = self._step_er_queue(action)
        elif self.task == "staff_allocation":
            reward, info = self._step_staff_allocation(action)

        self.t += 1
        self.done = self.t >= self.SHIFT_LENGTH
        return self.state, reward, self.done, info

    # -- Bed Allocation ----------------------------------------
    def _step_bed_allocation(self, action):
        free_beds = self.state["free_beds"]
        waiting_patients = self.state["waiting_patients"]
        reward = 0
        info = {}

        if action == "Admit":
            if free_beds > 0:
                reward = +10
                self.state["free_beds"] -= 1
                self.state["waiting_patients"] = max(0, waiting_patients - 1)
                self._discharge_timers.append(random.randint(1, 8))
                info["result"] = "Patient admitted successfully"
            else:
                reward = -10
                info["result"] = "Cannot admit -- no free beds"

        elif action == "Reject":
            reward = -5
            info["result"] = "Patient rejected"

        elif action == "Transfer":
            reward = +3
            self.state["waiting_patients"] = max(0, waiting_patients - 1)
            info["result"] = "Patient transferred to another facility"

        # World dynamics: beds free up, new patients arrive
        self._discharge_timers = [d - 1 for d in self._discharge_timers]
        discharged = sum(1 for d in self._discharge_timers if d <= 0)
        self._discharge_timers = [d for d in self._discharge_timers if d > 0]
        self.state["free_beds"] = min(
            self.total_beds, self.state["free_beds"] + discharged
        )

        new_arrivals = random.choices([0, 1, 2, 3], weights=[35, 35, 20, 10])[0]
        self.state["waiting_patients"] = min(
            60, self.state["waiting_patients"] + new_arrivals
        )

        info["new_arrivals"] = new_arrivals
        info["discharged"] = discharged

        return reward, info

    # -- ER Queue ------------------------------------------------
    def _step_er_queue(self, action):
        emergency_queue = self.state["emergency_queue"]
        normal_queue = self.state["normal_queue"]
        reward = 0
        info = {}

        if action == "Serve Emergency":
            if emergency_queue > 0:
                reward = +10
                self.state["emergency_queue"] -= 1
                info["result"] = "Emergency patient served"
            else:
                reward = -2
                info["result"] = "No emergency patients waiting"

        elif action == "Serve Normal":
            if normal_queue > 0:
                if emergency_queue > 0:
                    reward = -5
                    info["result"] = "Served normal while emergency waiting -- penalty"
                else:
                    reward = +5
                    info["result"] = "Normal patient served"
                self.state["normal_queue"] -= 1
            else:
                reward = -2
                info["result"] = "No normal patients waiting"

        # World dynamics: queues grow, neglect compounds
        new_emergency = random.choices([0, 1, 2], weights=[55, 35, 10])[0]
        new_normal = random.choices([0, 1, 2, 3], weights=[30, 35, 25, 10])[0]
        self.state["emergency_queue"] = min(
            20, self.state["emergency_queue"] + new_emergency
        )
        self.state["normal_queue"] = min(
            40, self.state["normal_queue"] + new_normal
        )

        if self.state["normal_queue"] > 15:
            reward -= 2
            info["overflow_penalty"] = True

        info["new_emergency"] = new_emergency
        info["new_normal"] = new_normal

        return reward, info

    # -- Staff Allocation ------------------------------------------
    def _step_staff_allocation(self, action):
        available_doctors = self.state["available_doctors"]
        patient_load = self.state["patient_load"]
        reward = 0
        info = {}

        if action == "Assign More Staff":
            if patient_load > 20:
                reward = +10
                self.state["available_doctors"] = min(20, available_doctors + 1)
                info["result"] = "More staff assigned -- high load justified"
            else:
                reward = -3
                self.state["available_doctors"] = min(20, available_doctors + 1)
                info["result"] = "Unnecessary staffing -- low patient load"

        elif action == "Keep Current":
            if 10 <= patient_load <= 20:
                reward = +5
                info["result"] = "Current staffing maintained -- balanced"
            else:
                reward = -2
                info["result"] = "Staffing level not optimal for current load"

        elif action == "Reduce Staff":
            if patient_load < 10:
                reward = +8
                self.state["available_doctors"] = max(1, available_doctors - 1)
                info["result"] = "Staff reduced -- low load, cost saved"
            else:
                reward = -10
                self.state["available_doctors"] = max(1, available_doctors - 1)
                info["result"] = "Dangerous -- reduced staff under high load"

        drift = random.choice([-4, -2, 0, 0, 2, 4, 6])
        doctors_now = self.state["available_doctors"]
        capacity = doctors_now * 4

        if patient_load > capacity:
            drift += 4
        self.state["patient_load"] = max(0, min(80, patient_load + drift))

        if doctors_now > 12:
            reward -= 1
            info["overstaff_cost"] = True

        info["drift"] = drift
        return reward, info

    def get_info(self):
        return {
            "task": self.task,
            "state_vars": self.state_vars,
            "actions": self.actions,
        }
