"""
UMORDA — Hospital Assistant Test Cases
Run this after training to verify the RAG+LLM+RL pipeline end-to-end
against the REAL Groq API and REAL trained Q-tables.

Covers:
  1. Normal single-task message
  2. Multi-task message (routes to 2+ tasks at once)
  3. Implausible input (should trigger clarification, not guess)
  4. Follow-up message (conversation memory — should ADD to prior state)
  5. Comparison mode (RAG-grounded vs ungrounded explanation)

Run: python test_assistant_cases.py
Requires: GROQ_API_KEY set, qtables/ already trained
"""

import os
from policy_retriever import PolicyRetriever
from hospital_assistant import process_task, TASK_LABELS
from llm_client import route_message
from state_manager import save_state, load_state


def reset_state():
    """Reset hospital_state.json to clean known values before each test."""
    fresh = {
        "bed_allocation":   {"free_beds": 8, "waiting_patients": 0, "last_updated": "test"},
        "er_queue":         {"emergency_queue": 0, "normal_queue": 0, "last_updated": "test"},
        "staff_allocation": {"available_doctors": 6, "patient_load": 0, "last_updated": "test"},
    }
    save_state(fresh)


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — cannot run live tests.")
        return

    retriever = PolicyRetriever("knowledge_base")
    history = []

    # ── Test 1: Normal single-task message ───────────────────────────────────
    section("TEST 1: Normal single-task message")
    reset_state()
    msg = "Beds are filling up fast, we've got a flood of patients coming in"
    print(f"  Input: \"{msg}\"")
    tasks = route_message(msg, history)
    print(f"  Routed to: {tasks}")
    assert "bed_allocation" in tasks, "FAIL: expected bed_allocation in routed tasks"
    for t in tasks:
        process_task(t, msg, history, retriever)
    history.append({"role": "user", "content": msg})

    # ── Test 2: Multi-task message ────────────────────────────────────────────
    section("TEST 2: Multi-task message (should route to 2 tasks)")
    reset_state()
    history = []
    msg = "The ER is backed up with emergencies and we're also short on doctors today"
    print(f"  Input: \"{msg}\"")
    tasks = route_message(msg, history)
    print(f"  Routed to: {tasks}")
    assert "er_queue" in tasks, "FAIL: expected er_queue in routed tasks"
    assert "staff_allocation" in tasks, "FAIL: expected staff_allocation in routed tasks"
    for t in tasks:
        process_task(t, msg, history, retriever)
    history.append({"role": "user", "content": msg})

    # ── Test 3: Implausible input → should ask for clarification ─────────────
    section("TEST 3: Implausible input (should trigger clarification)")
    reset_state()
    history = []
    msg = "We have 10000 patients waiting right now"
    print(f"  Input: \"{msg}\"")
    resolved = process_task("bed_allocation", msg, history, retriever)
    assert resolved is False, "FAIL: expected clarification (resolved=False), got resolved=True"
    print("  PASS: system asked for clarification instead of guessing.")

    # ── Test 4: Follow-up message — conversation memory ───────────────────────
    section("TEST 4: Follow-up message (memory — should ADD to prior state)")
    reset_state()
    history = []
    msg1 = "A lot of patients are waiting, maybe around 15"
    print(f"  Input 1: \"{msg1}\"")
    process_task("bed_allocation", msg1, history, retriever)
    history.append({"role": "user", "content": msg1})

    from state_manager import get_task_state
    state_after_1 = get_task_state("bed_allocation")
    print(f"  State after msg 1: {state_after_1}")

    msg2 = "A few more patients just arrived"
    print(f"  Input 2: \"{msg2}\"")
    process_task("bed_allocation", msg2, history, retriever)
    state_after_2 = get_task_state("bed_allocation")
    print(f"  State after msg 2: {state_after_2}")
    assert state_after_2["waiting_patients"] >= state_after_1["waiting_patients"], \
        "FAIL: follow-up should add to waiting_patients, not reset/decrease it"
    print("  PASS: follow-up correctly built on prior state.")

    # ── Test 5: Comparison mode ────────────────────────────────────────────────
    section("TEST 5: Comparison mode (grounded vs ungrounded)")
    reset_state()
    history = []
    msg = "We need more doctors, patient load is very high"
    print(f"  Input: \"{msg}\" (compare mode)")
    process_task("staff_allocation", msg, history, retriever, show_comparison=True)
    print("  PASS (visual check): compare the grounded vs ungrounded explanations above.")

    section("ALL TESTS COMPLETED")


if __name__ == "__main__":
    main()
