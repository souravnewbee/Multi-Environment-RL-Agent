import numpy as np
import random

class HospitalEnv:
    def __init__(self, task="bed_allocation"):
        self.task = task
        self.state = None
        self.done = False
        
        if self.task == "bed_allocation":
            self.state_vars  = ["free_beds", "waiting_patients"]
            self.actions     = ["Admit", "Reject", "Transfer"]
            self.n_actions   = 3
            
        elif self.task == "er_queue":
            self.state_vars  = ["emergency_queue", "normal_queue"]
            self.actions     = ["Serve Emergency", "Serve Normal"]
            self.n_actions   = 2
            
        elif self.task == "staff_allocation":
            self.state_vars  = ["available_doctors", "patient_load"]
            self.actions     = ["Assign More Staff", "Keep Current", "Reduce Staff"]
            self.n_actions   = 3
            
        else:
            raise ValueError(f"Unknown task: {task}")

    def reset(self):
        if self.task == "bed_allocation":
            self.state = {
                "free_beds":        random.randint(0, 20),
                "waiting_patients": random.randint(0, 30)
            }
        elif self.task == "er_queue":
            self.state = {
                "emergency_queue": random.randint(0, 10),
                "normal_queue":    random.randint(0, 20)
            }
        elif self.task == "staff_allocation":
            self.state = {
                "available_doctors": random.randint(1, 15),
                "patient_load":      random.randint(0, 50)
            }
        self.done = False
        return self.state

    def step(self, action_index):
        action  = self.actions[action_index]
        reward  = 0
        info    = {}

        # ── Bed Allocation ──────────────────────────────
        if self.task == "bed_allocation":
            free_beds        = self.state["free_beds"]
            waiting_patients = self.state["waiting_patients"]

            if action == "Admit":
                if free_beds > 0:
                    reward = +10
                    self.state["free_beds"]        -= 1
                    self.state["waiting_patients"] = max(0, waiting_patients - 1)
                    info["result"] = "Patient admitted successfully"
                else:
                    reward = -10
                    info["result"] = "Cannot admit — no free beds"

            elif action == "Reject":
                reward = -5
                info["result"] = "Patient rejected"

            elif action == "Transfer":
                reward = +3
                self.state["waiting_patients"] = max(0, waiting_patients - 1)
                info["result"] = "Patient transferred to another facility"

        # ── ER Queue ────────────────────────────────────
        elif self.task == "er_queue":
            emergency_queue = self.state["emergency_queue"]
            normal_queue    = self.state["normal_queue"]

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
                        info["result"] = "Served normal while emergency waiting — penalty"
                    else:
                        reward = +5
                        info["result"] = "Normal patient served"
                    self.state["normal_queue"] -= 1
                else:
                    reward = -2
                    info["result"] = "No normal patients waiting"

        # ── Staff Allocation ────────────────────────────
        elif self.task == "staff_allocation":
            available_doctors = self.state["available_doctors"]
            patient_load      = self.state["patient_load"]

            if action == "Assign More Staff":
                if patient_load > 20:
                    reward = +10
                    self.state["available_doctors"] += 1
                    info["result"] = "More staff assigned — high load justified"
                else:
                    reward = -3
                    info["result"] = "Unnecessary staffing — low patient load"

            elif action == "Keep Current":
                if 10 <= patient_load <= 20:
                    reward = +5
                    info["result"] = "Current staffing maintained — balanced"
                else:
                    reward = -2
                    info["result"] = "Staffing level not optimal for current load"

            elif action == "Reduce Staff":
                if patient_load < 10:
                    reward = +8
                    self.state["available_doctors"] = max(1, available_doctors - 1)
                    info["result"] = "Staff reduced — low load, cost saved"
                else:
                    reward = -10
                    info["result"] = "Dangerous — reduced staff under high load"

        self.done = True
        return self.state, reward, self.done, info

    def get_info(self):
        return {
            "task":       self.task,
            "state_vars": self.state_vars,
            "actions":    self.actions
        }
