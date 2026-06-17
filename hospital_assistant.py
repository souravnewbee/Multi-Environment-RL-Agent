"""
UMORDA — Hospital RAG + LLM + RL Assistant
Full pipeline: natural language in → RL decision → natural language out.

Flow:
  1. Load live hospital state (hospital_state.json)
  2. User describes situation in plain English
  3. LLM extracts/estimates structured state from message + known values
  4. Q-table (trained RL agent) picks the action
  5. Live state file is updated to reflect the action
  6. RAG retrieves the relevant policy passage
  7. LLM explains the decision in plain English, grounded in policy

Run: python hospital_assistant.py
Requires: GROQ_API_KEY environment variable set
"""

import os
import numpy as np

from state_manager import get_task_state, merge_new_arrivals, apply_action_effect
from policy_retriever import PolicyRetriever, build_query, TASK_SOURCE_MAP
from llm_client import extract_state, explain_decision

TASKS = {
    "1": ("bed_allocation",   "Bed Allocation",   ["Admit", "Transfer", "Reject"]),
    "2": ("er_queue",         "ER Queue",         ["Serve Emergency", "Serve Normal"]),
    "3": ("staff_allocation", "Staff Allocation", ["Assign More Staff", "Keep Current", "Reduce Staff"]),
}

# Internal reason hints — mirrors the logic baked into hospital_env.py rewards,
# used to give the explainer LLM something concrete to ground its explanation in.
def get_reason_hint(task, state, action):
    if task == "bed_allocation":
        beds = state["free_beds"]
        if action == "Admit":
            return f"{beds} beds available — sufficient capacity to admit safely"
        elif action == "Transfer":
            return f"only {beds} beds available — preserving capacity by transferring"
        else:
            return "rejection should rarely occur under current policy"

    elif task == "er_queue":
        if action == "Serve Emergency":
            return f"{state['emergency_queue']} emergency patients waiting — priority case"
        else:
            return f"emergency queue is empty — safe to serve normal queue ({state['normal_queue']} waiting)"

    elif task == "staff_allocation":
        load = state["patient_load"]
        if action == "Assign More Staff":
            return f"patient load is {load} — above safe staffing threshold"
        elif action == "Reduce Staff":
            return f"patient load is {load} — low enough to reduce cost safely"
        else:
            return f"patient load is {load} — within balanced staffing range"


def discretize(state, task):
    if task == "bed_allocation":
        beds     = min(state["free_beds"], 20) // 5
        patients = min(state["waiting_patients"], 30) // 10
        return (beds, patients)
    elif task == "er_queue":
        emergency = min(state["emergency_queue"], 10) // 3
        normal    = min(state["normal_queue"], 20) // 5
        return (emergency, normal)
    elif task == "staff_allocation":
        doctors = min(state["available_doctors"], 15) // 5
        load    = min(state["patient_load"], 50) // 15
        return (doctors, load)


def load_qtable(task):
    path = f"qtables/hospital_{task}.npy"
    if not os.path.exists(path):
        print(f"\n  Q-table not found at {path}. Run train_hospital.py first.\n")
        return None
    return np.load(path)


def run_pipeline(task, task_label, actions, user_message, retriever):
    print(f"\n  ── Processing ({task_label}) ──────────────────")

    # Step 1: known live state
    known_state = get_task_state(task)
    print(f"  Known state before  : {known_state}")

    # Step 2: LLM extracts/estimates updated state from message
    state = extract_state(task, user_message, known_state)
    merge_new_arrivals(task, state)
    print(f"  Extracted state      : {state}")

    # Step 3: Q-table decision
    Q = load_qtable(task)
    if Q is None:
        return
    s          = discretize(state, task)
    action_idx = int(np.argmax(Q[s]))
    action     = actions[action_idx]
    print(f"  RL Decision           : {action}")

    # Step 4: update live state to reflect the action taken
    new_state = apply_action_effect(task, action)
    print(f"  Updated live state    : {new_state}")

    # Step 5: RAG retrieval
    query   = build_query(task, state, action)
    chunks  = retriever.retrieve(query, top_k=2, source_filter=TASK_SOURCE_MAP[task])

    # Step 6: LLM explanation
    reason_hint   = get_reason_hint(task, state, action)
    explanation   = explain_decision(task, state, action, reason_hint, chunks)

    print(f"\n  ── Recommendation ─────────────────────")
    print(f"  Action      : {action}")
    print(f"  Explanation : {explanation}\n")


def main():
    print("\n")
    print("*" * 54)
    print("*   UMORDA — HOSPITAL RAG + LLM + RL ASSISTANT  *")
    print("*" * 54)

    if not os.environ.get("GROQ_API_KEY"):
        print("\n  WARNING: GROQ_API_KEY not set.")
        print("  Get a free key at https://console.groq.com")
        print("  Then run: setx GROQ_API_KEY \"your-key\"  (Windows, new terminal needed after)")
        print("  or:       export GROQ_API_KEY=\"your-key\"  (Mac/Linux)\n")
        return

    retriever = PolicyRetriever("knowledge_base")
    print(f"\n  Knowledge base loaded: {len(retriever.chunks)} policy chunks.\n")

    while True:
        print("\n  Select a task:")
        print("  1. Bed Allocation")
        print("  2. ER Queue Management")
        print("  3. Staff Allocation")
        print("  4. Exit")

        choice = input("\n  Enter choice (1/2/3/4): ").strip()

        if choice in TASKS:
            task, label, actions = TASKS[choice]
            user_message = input(f"\n  Describe the {label} situation in your own words: ").strip()
            run_pipeline(task, label, actions, user_message, retriever)
        elif choice == "4":
            print("\n  Exiting. Goodbye!\n")
            break
        else:
            print("\n  Invalid choice. Please enter 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()