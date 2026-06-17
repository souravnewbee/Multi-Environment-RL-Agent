"""
UMORDA — Live Hospital State Manager
Tracks the hospital's real current numbers (free beds, doctors on duty, etc.)
in a small JSON file that updates as the RL agent acts.

This is intentionally simple — no database needed. For a class project this
file IS the hospital's live status board.
"""

import json
import os
from datetime import datetime

STATE_FILE = "hospital_state.json"


def load_state(path=STATE_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Make sure hospital_state.json exists in the project root."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state, path=STATE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_task_state(task, path=STATE_FILE):
    """Return just the live numbers for one task, e.g. {'free_beds': 8, 'waiting_patients': 0}"""
    full = load_state(path)
    if task not in full:
        raise ValueError(f"Unknown task '{task}' in {path}")
    task_state = dict(full[task])
    task_state.pop("last_updated", None)
    return task_state


def apply_action_effect(task, action, path=STATE_FILE):
    """
    Update the live state file based on the action the RL agent chose.
    This mirrors the same transition logic as hospital_env.py so the
    'real' hospital numbers move consistently with what the agent learned.
    """
    full  = load_state(path)
    state = full[task]

    if task == "bed_allocation":
        if action == "Admit" and state["free_beds"] > 0:
            state["free_beds"]        = max(0, state["free_beds"] - 1)
            state["waiting_patients"] = max(0, state["waiting_patients"] - 1)
        elif action == "Transfer":
            state["waiting_patients"] = max(0, state["waiting_patients"] - 1)
        # Reject: no state change (and policy says it shouldn't happen anyway)

    elif task == "er_queue":
        if action == "Serve Emergency" and state["emergency_queue"] > 0:
            state["emergency_queue"] -= 1
        elif action == "Serve Normal" and state["normal_queue"] > 0:
            state["normal_queue"] -= 1

    elif task == "staff_allocation":
        if action == "Assign More Staff":
            state["available_doctors"] = min(15, state["available_doctors"] + 1)
        elif action == "Reduce Staff":
            state["available_doctors"] = max(1, state["available_doctors"] - 1)
        # Keep Current: no change

    state["last_updated"] = datetime.now().isoformat(timespec="seconds")
    full[task] = state
    save_state(full, path)
    return state


def merge_new_arrivals(task, extracted_fields, path=STATE_FILE):
    """
    Merge LLM-extracted dynamic fields (e.g. new waiting_patients mentioned
    by the user) into the live state, then persist it.

    extracted_fields: dict of only the fields the LLM extracted/estimated,
                       e.g. {"waiting_patients": 12}
    """
    full  = load_state(path)
    state = full[task]
    state.update(extracted_fields)
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")
    full[task] = state
    save_state(full, path)
    return state


if __name__ == "__main__":
    # Smoke test
    print("Initial state:", get_task_state("bed_allocation"))
    merge_new_arrivals("bed_allocation", {"waiting_patients": 12})
    print("After merge   :", get_task_state("bed_allocation"))
    apply_action_effect("bed_allocation", "Admit")
    print("After Admit   :", get_task_state("bed_allocation"))
    