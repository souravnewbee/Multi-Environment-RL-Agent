"""
UMORDA — Main Entry Point
File: main.py

Ties the three layers together for every domain in one CLI:

    1. route_message()   -> which task(s) is the user describing?
    2. extract_state()   -> pull numeric state out of free text
       (per-task state is remembered across turns; if the LLM needs
       clarification, we pause on that task and resume on the next reply)
    3. trained Q-table    -> optimal action for that state (with any
       domain-specific hard safety override applied)
    4. RAG + explain_decision() -> plain-English explanation grounded in
       the matching knowledge_base/*.md policy file

Covers:
    Hospital     : bed_allocation, er_queue, staff_allocation
    Traffic      : intersection, pedestrian, parking
    Energy       : solar_scheduling, battery_management, grid_interaction
    Finance      : trading, savings, budget
    Agriculture  : soil_preparation, irrigation, pest_control
                   (routes + explains, but the Q-table isn't trained yet --
                   see training/train_agriculture.py)

Requires: GROQ_API_KEY (or LLM_BACKEND=ollama) -- see llm_client.py
Run:      python main.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.hospital_env import HospitalEnv
from environments.traffic_env import TrafficEnv
from environments.energy_env import EnergyEnv
from agents.hospital_agent import discretize as hospital_discretize

from llm_client import (
    route_message, extract_state, explain_decision, explain_ungrounded,
    TASK_FIELD_SPECS,
)
from policy_retriever import PolicyRetriever, build_query, TASK_SOURCE_MAP

# All trained Q-tables live in one SQLite DB (qtables/qtables.db), written by
# every training/train_*.py script and every agents/*_agent.py class. This is
# the single source of truth for decisions below -- do not fall back to
# np.load() on standalone .npy files, none of the training scripts write those.
from qtable_store import load_qtable as db_load_qtable

try:
    from agents.finance_agent import FinanceAgent
    HAS_FINANCE_AGENT = True
except ImportError:
    HAS_FINANCE_AGENT = False

try:
    from agents.agriculture_agent import discretize as agriculture_discretize
    from environments.agriculture_env import AgricultureEnv
    HAS_AGRICULTURE_AGENT = True
except ImportError:
    HAS_AGRICULTURE_AGENT = False

MAX_HISTORY  = 12   # conversation memory length, in messages

# =============================================================================
# Task -> domain routing
# =============================================================================
TASK_DOMAIN = {
    "bed_allocation":     "hospital",
    "er_queue":           "hospital",
    "staff_allocation":   "hospital",
    "intersection":       "traffic",
    "pedestrian":         "traffic",
    "parking":            "traffic",
    "solar_scheduling":   "energy",
    "battery_management": "energy",
    "grid_interaction":   "energy",
    "trading":            "finance",
    "savings":            "finance",
    "budget":              "finance",
    "soil_preparation":   "agriculture",
    "irrigation":         "agriculture",
    "pest_control":       "agriculture",
}

TASK_LABELS = {t: t.replace("_", " ").title() for t in TASK_DOMAIN}

# Field order per task, driven by TASK_FIELD_SPECS (dict order == env order
# for every domain wired up so far -- see chat notes / README for the check).
def _state_order(task):
    return list(TASK_FIELD_SPECS[task].keys())


# Reasonable starting values so a first message doesn't need to specify
# every field. Extraction only overwrites the fields the user mentions.
DEFAULT_STATE = {
    # Hospital
    "bed_allocation":   {"free_beds": 8, "waiting_patients": 5},
    "er_queue":         {"emergency_queue": 0, "normal_queue": 3},
    "staff_allocation": {"available_doctors": 6, "patient_load": 15},
    # Traffic
    "intersection": {"cars_NS": 2, "cars_EW": 2, "current_phase": 0,
                      "phase_elapsed": 0, "wait_NS": 0, "wait_EW": 0},
    "pedestrian":   {"peds": 0, "vehs": 0, "ped_wait": 0,
                      "veh_wait": 0, "phase": 1, "elapsed": 0},
    "parking":      {"spots": 10, "incoming": 0, "queue_wait": 0, "occupancy": 0},
    # Energy
    "solar_scheduling":   {"solar_output": 0, "home_consumption": 3,
                            "battery_level": 5, "time_of_day": 1},
    "battery_management": {"battery_level": 5, "solar_output": 0,
                            "grid_price": 1, "home_consumption": 3},
    "grid_interaction":   {"grid_price": 1, "solar_surplus": 0,
                            "battery_level": 5, "home_consumption": 3},
    # Finance
    "trading": {"price_trend": 0, "shares_held": 0, "cash": 1000,
                "portfolio_value": 1000},
    "savings": {"monthly_income": 50, "current_savings": 200,
                "expenses": 40, "months_remaining": 12},
    "budget":  {"total_budget": 800, "amount_spent": 0,
                "urgent_requests": 0, "departments_remaining": 5},
    # Agriculture
    "soil_preparation": {"soil_ph": 6, "organic_matter": 30,
                          "drainage_quality": 30, "days_remaining": 25},
    "irrigation":        {"water_reservoir": 60, "crop_stress": 20,
                           "rainfall_trend": 0, "days_remaining": 40},
    "pest_control":       {"total_resource": 750, "resource_used": 0,
                            "urgent_outbreaks": 0, "plots_remaining": 6},
}


# =============================================================================
# Per-domain decision functions
# Each returns (action_name, error_message) -- exactly one is None.
# All read from qtables/qtables.db via qtable_store.load_qtable(), which
# raises FileNotFoundError with a helpful message if that key was never
# trained -- we just surface that message as-is.
# =============================================================================
def decide_hospital(task, state):
    order = _state_order(task)
    obs   = [state[k] for k in order]

    try:
        Q = db_load_qtable(f"hospital_{task}")
    except FileNotFoundError as e:
        return None, str(e)

    try:
        s = hospital_discretize(obs, task)
        action_idx = int(np.argmax(Q[s]))
    except (IndexError, KeyError) as e:
        return None, (f"Discretization/Q-table shape mismatch for hospital/{task} ({e}). "
                       f"Retrain with training/train_hospital.py.")

    action = HospitalEnv(task=task).actions[action_idx]
    return action, None


def decide_traffic(task, state):
    env   = TrafficEnv(task=task)
    order = _state_order(task)
    raw   = [state[k] for k in order]

    encoded        = env._encode_state(raw)
    env._raw_state = raw

    try:
        Q = db_load_qtable(f"traffic_{task}")
    except FileNotFoundError as e:
        return None, str(e)

    if encoded >= Q.shape[0]:
        return None, "State index out of range for saved Q-table (shape mismatch). Retrain needed."

    q_action     = int(np.argmax(Q[encoded]))
    final_action = env._safety_override(q_action)   # hard safety guarantee
    return env.cfg["action_meanings"][final_action], None


def decide_energy(task, state):
    env   = EnergyEnv(task=task)
    order = _state_order(task)
    raw   = [state[k] for k in order]
    encoded = env._encode_state(raw)

    try:
        Q = db_load_qtable(f"energy_{task}")
    except FileNotFoundError as e:
        return None, str(e)

    if encoded >= Q.shape[0]:
        return None, "State index out of range for saved Q-table (shape mismatch). Retrain needed."

    action_idx = int(np.argmax(Q[encoded]))
    return env.cfg["action_meanings"][action_idx], None


def decide_finance(task, state):
    if not HAS_FINANCE_AGENT:
        return None, "agents/finance_agent.py not found."
    try:
        agent = FinanceAgent(task=task)
    except FileNotFoundError as e:
        return None, str(e)
    _, _, action_name = agent.get_action(state)
    return action_name, None


def decide_agriculture(task, state):
    if not HAS_AGRICULTURE_AGENT:
        return None, "agents/agriculture_agent.py failed to import -- check for errors in that file."

    order = _state_order(task)
    obs   = [state[k] for k in order]

    try:
        Q = db_load_qtable(f"agriculture_{task}")
    except FileNotFoundError as e:
        return None, str(e)

    try:
        s = agriculture_discretize(obs, task)
        action_idx = int(np.argmax(Q[s]))
    except (IndexError, KeyError) as e:
        return None, (f"Discretization/Q-table shape mismatch for agriculture/{task} ({e}). "
                       f"Retrain with training/train_agriculture.py.")

    action = AgricultureEnv(task=task).actions[action_idx]
    return action, None


DECIDERS = {
    "hospital":    decide_hospital,
    "traffic":     decide_traffic,
    "energy":      decide_energy,
    "finance":     decide_finance,
    "agriculture": decide_agriculture,
}


def get_decision(task, state):
    domain = TASK_DOMAIN.get(task)
    if domain not in DECIDERS:
        return None, f"Unknown domain for task '{task}'."
    return DECIDERS[domain](task, state)


# =============================================================================
# Explanation (RAG-grounded, falls back to ungrounded if retrieval is empty)
# =============================================================================
def get_explanation(retriever, task, state, action, show_comparison=False):
    reason_hint = f"Selected as the optimal action by the trained Q-learning policy for '{task}'."

    chunks = []
    if task in TASK_SOURCE_MAP:
        query  = build_query(task, state, action)
        chunks = retriever.retrieve(query, top_k=2, source_filter=TASK_SOURCE_MAP[task])

    if chunks:
        explanation = explain_decision(task, state, action, reason_hint, chunks)
    else:
        explanation = explain_ungrounded(task, state, action, reason_hint)

    comparison = None
    if show_comparison and chunks:
        comparison = explain_ungrounded(task, state, action, reason_hint)

    return explanation, chunks, comparison


# =============================================================================
# Per-turn processing for one routed task
# =============================================================================
def process_task(task, user_message, known_state, conversation_history,
                  retriever, show_comparison=False):
    """
    Returns True if the task was fully resolved this turn, False if the LLM
    needs clarification (in which case the caller should pause on this task
    and resume with the user's next reply).
    """
    label = TASK_LABELS[task]
    print(f"\n  -- {label} " + "-" * max(1, 40 - len(label)))

    extraction = extract_state(task, user_message, known_state[task], conversation_history)

    if extraction["needs_clarification"]:
        question = extraction["clarification_question"] or "Could you clarify the numbers involved?"
        print(f"  \U0001F914 {question}")
        return False

    known_state[task] = extraction["state"]
    if extraction["notes"]:
        print(f"  (extracted: {known_state[task]}  |  {extraction['notes']})")
    else:
        print(f"  (extracted: {known_state[task]})")

    action, error = get_decision(task, known_state[task])
    if error:
        print(f"  [!] {error}")
        return True   # nothing more to do for this task this turn

    explanation, chunks, comparison = get_explanation(
        retriever, task, known_state[task], action, show_comparison
    )

    if chunks:
        print(f"  \U0001F4CB Retrieved policy ({chunks[0]['source']}, relevance={chunks[0]['score']:.2f}):")
        print(f"     \"{chunks[0]['text'].splitlines()[0]}\"")

    print(f"\n  \u2705 Action      : {action}")
    print(f"  \U0001F4AC Explanation : {explanation}")

    if comparison:
        print(f"\n  \u26A0\uFE0F  Without RAG (for comparison):")
        print(f"     {comparison}")

    return True


# =============================================================================
# CLI
# =============================================================================
def main():
    print("\n" + "*" * 60)
    print("*   UMORDA -- Universal Multi-Objective RL Decision Agent   *")
    print("*" * 60)

    if not os.environ.get("GROQ_API_KEY") and os.environ.get("LLM_BACKEND", "groq") != "ollama":
        print("\n  WARNING: GROQ_API_KEY not set.")
        print("  Get a free key at https://console.groq.com")
        print("  Then run: export GROQ_API_KEY=\"your-key\"")
        print("  (or set LLM_BACKEND=ollama to use a local model instead)\n")
        return

    try:
        retriever = PolicyRetriever("knowledge_base")
    except FileNotFoundError as e:
        print(f"\n  [!] {e}\n")
        return
    print(f"\n  Knowledge base loaded: {len(retriever.chunks)} policy chunks.")

    print("  Describe any situation in plain English -- hospital beds, an ER queue,")
    print("  a traffic intersection, solar/battery/grid, trading, savings, a budget,")
    print("  or soil/irrigation/pest decisions. Type 'exit' to quit.")
    print("  Prefix a message with 'compare ' to also see an ungrounded explanation.")
    print("  Type 'state' to see everything currently known, 'reset' to clear it.\n")

    known_state = {t: dict(DEFAULT_STATE[t]) for t in DEFAULT_STATE}
    conversation_history = []
    pending_task = None   # set while waiting on a clarification reply

    while True:
        user_message = input("\n  You: ").strip()
        if not user_message:
            continue
        if user_message.lower() in ("exit", "quit"):
            print("\n  Exiting. Goodbye!\n")
            break
        if user_message.lower() == "state":
            for task, state in known_state.items():
                print(f"    [{task}]: {state}")
            continue
        if user_message.lower() == "reset":
            known_state = {t: dict(DEFAULT_STATE[t]) for t in DEFAULT_STATE}
            conversation_history = []
            pending_task = None
            print("  All state reset. \u2705")
            continue

        show_comparison = False
        if user_message.lower().startswith("compare "):
            show_comparison = True
            user_message = user_message[len("compare "):].strip()

        conversation_history.append({"role": "user", "content": user_message})
        conversation_history[:] = conversation_history[-MAX_HISTORY:]

        if pending_task:
            resolved = process_task(pending_task, user_message, known_state,
                                     conversation_history, retriever, show_comparison)
            if resolved:
                pending_task = None
            continue

        tasks = route_message(user_message, conversation_history)
        if not tasks:
            print("\n  Couldn't confidently match this to a known task. Try mentioning")
            print("  beds/ER/staffing, an intersection/crossing/parking lot, solar/battery/grid,")
            print("  trading/savings/budget, or soil/irrigation/pests specifically.")
            continue

        for task in tasks:
            resolved = process_task(task, user_message, known_state,
                                     conversation_history, retriever, show_comparison)
            if not resolved:
                pending_task = task
                break

        conversation_history.append({
            "role": "assistant",
            "content": f"Processed: {', '.join(tasks)}",
        })


if __name__ == "__main__":
    main()
