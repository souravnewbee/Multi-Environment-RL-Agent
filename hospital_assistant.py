"""
UMORDA — Hospital RAG + LLM + RL Assistant (v2)
Full conversational pipeline: free-form natural language in -> RL decision(s) out.

New in this version:
  - No menu. User just describes the situation in plain English.
  - LLM Router decides which task(s) the message concerns (can be multiple).
  - Conversation memory: follow-up messages build on prior context
    ("a few more patients arrived" adds to, doesn't replace, known state).
  - Clarification: if input is implausible or too vague, the system asks
    a natural follow-up question instead of guessing.
  - RAG transparency: the retrieved policy passage is shown, not hidden.
  - Optional grounded-vs-ungrounded comparison, toggled per-query.

Run: python hospital_assistant.py
Requires: GROQ_API_KEY environment variable set
"""

import os
import numpy as np

from state_manager import get_task_state, merge_new_arrivals, apply_action_effect
from policy_retriever import PolicyRetriever, build_query, TASK_SOURCE_MAP
from llm_client import route_message, extract_state, explain_decision, explain_ungrounded

TASK_LABELS = {
    "bed_allocation":   "Bed Allocation",
    "er_queue":         "ER Queue",
    "staff_allocation": "Staff Allocation",
}
TASK_ACTIONS = {
    "bed_allocation":   ["Admit", "Transfer", "Reject"],
    "er_queue":         ["Serve Emergency", "Serve Normal"],
    "staff_allocation": ["Assign More Staff", "Keep Current", "Reduce Staff"],
}

MAX_HISTORY = 12   # cap conversation memory length (messages, not tokens)


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


def process_task(task, user_message, conversation_history, retriever, show_comparison=False):
    """Run the full pipeline for one routed task. Returns False if clarification was needed (paused)."""
    label   = TASK_LABELS[task]
    actions = TASK_ACTIONS[task]

    print(f"\n  ── {label} ──────────────────────────────")

    # Known live state
    known_state = get_task_state(task)

    # LLM extraction with conversation memory + plausibility check
    extraction = extract_state(task, user_message, known_state, conversation_history)

    if extraction["needs_clarification"]:
        question = extraction["clarification_question"] or "Could you clarify the numbers involved?"
        print(f"  🤔 {question}")
        return False   # signal: paused, waiting on user

    state = extraction["state"]
    if extraction["notes"]:
        print(f"  (extracted: {state}  |  {extraction['notes']})")
    else:
        print(f"  (extracted: {state})")

    merge_new_arrivals(task, state)

    # Q-table decision
    Q = load_qtable(task)
    if Q is None:
        return True
    s          = discretize(state, task)
    action_idx = int(np.argmax(Q[s]))
    action     = actions[action_idx]

    # Update live state
    apply_action_effect(task, action)

    # RAG retrieval — shown transparently
    query  = build_query(task, state, action)
    chunks = retriever.retrieve(query, top_k=2, source_filter=TASK_SOURCE_MAP[task])
    reason_hint = get_reason_hint(task, state, action)

    print(f"\n  📋 Retrieved policy ({chunks[0]['source']}, relevance={chunks[0]['score']:.2f}):")
    snippet = chunks[0]["text"].split("\n")[0]   # header line
    print(f"     \"{snippet}\"")

    explanation = explain_decision(task, state, action, reason_hint, chunks)

    print(f"\n  ✅ Action      : {action}")
    print(f"  💬 Explanation : {explanation}")

    if show_comparison:
        ungrounded = explain_ungrounded(task, state, action, reason_hint)
        print(f"\n  ⚠️  Without RAG (for comparison, may be generic/ungrounded):")
        print(f"     {ungrounded}")

    return True


def main():
    print("\n")
    print("*" * 56)
    print("*   UMORDA — HOSPITAL RAG + LLM + RL ASSISTANT (v2)   *")
    print("*" * 56)

    if not os.environ.get("GROQ_API_KEY"):
        print("\n  WARNING: GROQ_API_KEY not set.")
        print("  Get a free key at https://console.groq.com")
        print("  Then run: setx GROQ_API_KEY \"your-key\"  (Windows, new terminal needed after)")
        print("  or:       export GROQ_API_KEY=\"your-key\"  (Mac/Linux)\n")
        return

    retriever = PolicyRetriever("knowledge_base")
    print(f"\n  Knowledge base loaded: {len(retriever.chunks)} policy chunks.")
    print("  Just describe what's happening in plain English. Type 'exit' to quit.")
    print("  Type 'compare' before a message to also see an ungrounded (no-RAG) explanation.\n")

    conversation_history = []
    pending_task = None   # set when a clarification question is waiting on a reply

    while True:
        user_message = input("\n  You: ").strip()
        if not user_message:
            continue
        if user_message.lower() in ("exit", "quit"):
            print("\n  Exiting. Goodbye!\n")
            break

        show_comparison = False
        if user_message.lower().startswith("compare "):
            show_comparison = True
            user_message = user_message[len("compare "):].strip()

        conversation_history.append({"role": "user", "content": user_message})
        conversation_history[:] = conversation_history[-MAX_HISTORY:]

        # If we're resuming after a clarification question, re-process the same task
        if pending_task:
            resolved = process_task(pending_task, user_message, conversation_history, retriever, show_comparison)
            if resolved:
                pending_task = None
            continue

        # Route the message to relevant task(s)
        tasks = route_message(user_message, conversation_history)

        if not tasks:
            print("\n  This doesn't seem related to bed allocation, ER queue, or staffing.")
            print("  Could you describe a hospital resource situation?")
            continue

        for task in tasks:
            resolved = process_task(task, user_message, conversation_history, retriever, show_comparison)
            if not resolved:
                pending_task = task   # wait for clarification reply before continuing
                break

        conversation_history.append({
            "role": "assistant",
            "content": f"Processed: {', '.join(tasks)}",
        })


if __name__ == "__main__":
    main()