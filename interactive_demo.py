"""
UMORDA — Interactive Live Demo
File: interactive_demo.py

Type a natural-language message, and the REAL pipeline runs end-to-end:

    1. route_message()   -> which task is this about?
    2. extract_state()   -> pull numeric state out of your sentence
    3. trained Q-table   -> what does the agent actually decide?
    4. explain_decision()-> plain-English explanation of that decision

All three LLM steps (route/extract/explain) go through whichever backend
is set in llm_client.py's LLM_BACKEND (Ollama 7B or Groq 70B) — this file
does not care which one is active, it just uses llm_client as-is.

Run from your project root (same folder as llm_client.py):

    Windows CMD:
        set LLM_BACKEND=ollama
        python interactive_demo.py

    PowerShell:
        $env:LLM_BACKEND="ollama"
        python interactive_demo.py

Type 'exit' or 'quit' to stop.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import llm_client


# ─────────────────────────────────────────────────────────────────────────
# Flexible imports — works whether your envs/agents live in
# environments/ + agents/ subfolders, or flat in the project root.
# ─────────────────────────────────────────────────────────────────────────
def _flex_import(module_name):
    """Try `environments.X` / `agents.X`, fall back to flat `X`."""
    for prefix in ("environments.", "agents.", ""):
        try:
            return __import__(prefix + module_name, fromlist=["*"])
        except ImportError:
            continue
    raise ImportError(f"Could not find module '{module_name}' in environments/, agents/, or root.")


energy_env_mod    = _flex_import("energy_env")
traffic_env_mod   = _flex_import("traffic_env")
hospital_env_mod  = _flex_import("hospital_env")
finance_agent_mod = _flex_import("finance_agent")

try:
    hospital_agent_mod = _flex_import("hospital_agent")
    HAS_HOSPITAL_AGENT = True
except ImportError:
    HAS_HOSPITAL_AGENT = False


# ─────────────────────────────────────────────────────────────────────────
# Task → domain mapping
# ─────────────────────────────────────────────────────────────────────────
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
    "budget":             "finance",
    "soil_preparation":   "agriculture",
    "irrigation":         "agriculture",
    "pest_control":       "agriculture",
}

HOSPITAL_STATE_VARS = {
    "bed_allocation":   ["free_beds", "waiting_patients"],
    "er_queue":         ["emergency_queue", "normal_queue"],
    "staff_allocation": ["available_doctors", "patient_load"],
}


def _find_qtable(candidates):
    """Return the first existing path from a list of candidate qtable paths."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ─────────────────────────────────────────────────────────────────────────
# Per-domain: get trained agent's decision for a given state dict
# Returns (action_name, error_message)  -- one of them will be None
# ─────────────────────────────────────────────────────────────────────────
def decide_energy(task, state):
    cfg  = energy_env_mod.EnergyEnv.TASK_CONFIG[task]
    bins = cfg["state_bins"]
    raw  = [int(np.clip(state.get(v, 0), 0, bins[i] - 1))
            for i, v in enumerate(cfg["state_vars"])]

    idx, multiplier = 0, 1
    for i in reversed(range(len(raw))):
        idx += raw[i] * multiplier
        multiplier *= bins[i]

    path = _find_qtable([
        f"qtables/energy_{task}_qtable.npy",
        f"qtables/energy_{task}.npy",
    ])
    if not path:
        return None, f"No trained Q-table found for energy/{task}. Run train_energy.py first."

    Q = np.load(path)
    if idx >= Q.shape[0]:
        return None, f"State index out of range for saved Q-table (shape mismatch). Retrain needed."
    action_idx = int(np.argmax(Q[idx]))
    return cfg["action_meanings"][action_idx], None


def decide_traffic(task, state):
    cfg  = traffic_env_mod.TrafficEnv.TASK_CONFIG[task]
    bins = cfg["state_bins"]
    raw  = [int(np.clip(state.get(v, 0), 0, bins[i] - 1))
            for i, v in enumerate(cfg["state_vars"])]

    idx, multiplier = 0, 1
    for i in reversed(range(len(raw))):
        idx += raw[i] * multiplier
        multiplier *= bins[i]

    path = _find_qtable([
        f"qtables/traffic_{task}_qtable.npy",
        f"qtables/traffic_{task}.npy",
    ])
    if not path:
        return None, f"No trained Q-table found for traffic/{task}. Run train_traffic.py first."

    Q = np.load(path)
    if idx >= Q.shape[0]:
        return None, f"State index out of range for saved Q-table (shape mismatch). Retrain needed."
    action_idx = int(np.argmax(Q[idx]))
    return cfg["action_meanings"][action_idx], None


def decide_hospital(task, state):
    if not HAS_HOSPITAL_AGENT:
        return None, "hospital_agent.py not found — cannot discretize state."

    state_vars = HOSPITAL_STATE_VARS[task]
    obs = [state.get(v, 0) for v in state_vars]

    path = _find_qtable([
        f"qtables/hospital_{task}.npy",
        f"qtables/hospital_{task}_qtable.npy",
    ])
    if not path:
        return None, f"No trained Q-table found for hospital/{task}. Run train_hospital.py first."

    Q = np.load(path)
    try:
        s = hospital_agent_mod.discretize(obs, task)
        action_idx = int(np.argmax(Q[s]))
    except (IndexError, KeyError) as e:
        return None, (f"Discretization/Q-table shape mismatch for hospital/{task} "
                       f"({e}). The saved Q-table may not match the current "
                       f"discretize() bins — retrain with train_hospital.py.")

    env = hospital_env_mod.HospitalEnv(task=task)
    return env.actions[action_idx], None


def decide_finance(task, state):
    try:
        agent = finance_agent_mod.FinanceAgent(task=task)
    except FileNotFoundError as e:
        return None, str(e)
    action_idx, q_values, action_name = agent.get_action(state)
    return action_name, None


def get_agent_decision(task, state):
    domain = TASK_DOMAIN.get(task)
    if domain == "energy":
        return decide_energy(task, state)
    elif domain == "traffic":
        return decide_traffic(task, state)
    elif domain == "hospital":
        return decide_hospital(task, state)
    elif domain == "finance":
        return decide_finance(task, state)
    elif domain == "agriculture":
        return None, "No trained Q-learning agent wired up yet for agriculture tasks."
    return None, f"Unknown domain for task '{task}'."


# ─────────────────────────────────────────────────────────────────────────
# Optional: policy retrieval (RAG). Falls back to ungrounded explanation
# if knowledge_base/ or sklearn isn't available.
# ─────────────────────────────────────────────────────────────────────────
try:
    from policy_retriever import PolicyRetriever, build_query, TASK_SOURCE_MAP
    _retriever = PolicyRetriever("knowledge_base")
    HAS_POLICY = True
except Exception as e:
    HAS_POLICY = False
    _POLICY_ERROR = str(e)


def get_explanation(task, state, action, reason_hint):
    if HAS_POLICY and task in TASK_SOURCE_MAP:
        try:
            query  = build_query(task, state, action)
            chunks = _retriever.retrieve(query, top_k=2,
                                         source_filter=TASK_SOURCE_MAP.get(task))
            if chunks:
                return llm_client.explain_decision(task, state, action, reason_hint, chunks)
        except Exception:
            pass
    return llm_client.explain_ungrounded(task, state, action, reason_hint)


# ─────────────────────────────────────────────────────────────────────────
# Main interactive loop
# ─────────────────────────────────────────────────────────────────────────
def main():
    backend = llm_client.LLM_BACKEND
    model   = llm_client.OLLAMA_MODEL if backend == "ollama" else llm_client.MODEL

    print("\n" + "#" * 62)
    print("#   UMORDA — INTERACTIVE LIVE DEMO")
    print(f"#   LLM backend: {backend}  (model={model})")
    print(f"#   Policy retrieval (RAG): {'ON' if HAS_POLICY else 'OFF - ' + _POLICY_ERROR if not HAS_POLICY else 'ON'}")
    print("#" * 62)
    print("\nType a message describing a situation (e.g. 'ER is packed with")
    print("emergency patients' or 'battery is almost full and price is high').")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not msg:
            continue
        if msg.lower() in ("exit", "quit"):
            print("Bye.")
            break

        # 1. Route
        tasks = llm_client.route_message(msg)
        if not tasks:
            print("  [!] Could not confidently route this message to a task. Try rephrasing.\n")
            continue
        task = tasks[0]
        print(f"  → Routed to: {task}  (domain: {TASK_DOMAIN.get(task, 'unknown')})")

        # 2. Extract
        known_state = {k: 0 for k in llm_client.TASK_FIELD_SPECS[task]}
        extraction  = llm_client.extract_state(task, msg, known_state)
        state       = extraction["state"]
        print(f"  → Extracted state: {state}")
        if extraction["notes"]:
            print(f"  → Notes: {extraction['notes']}")

        if extraction["needs_clarification"]:
            print(f"  → Needs clarification: {extraction['clarification_question']}\n")
            continue

        # 3. Trained agent decision
        action, error = get_agent_decision(task, state)
        if error:
            print(f"  → [!] {error}\n")
            continue
        print(f"  → Agent decision: {action}")

        # 4. Explain
        reason_hint = f"Selected as the optimal action by the trained Q-learning policy for '{task}'."
        explanation = get_explanation(task, state, action, reason_hint)
        print(f"\n  Explanation:\n  {explanation}\n")


if __name__ == "__main__":
    main()
