"""
UMORDA — Full System Check
File: check_system.py  (place at project root, next to main.py)

Runs every layer of the pipeline independently and prints exactly what
passed/failed at each stage, so you can see WHERE something breaks instead
of just "the browser showed an error."

Layers checked, in order:
    0. Python environment + required packages
    1. LLM backend config (GROQ_API_KEY / LLM_BACKEND)
    2. Q-table database (qtables/qtables.db) — what's actually trained
    3. Knowledge base (RAG policy retriever)
    4. Live hospital state (hospital_state.json)
    5. FULL PIPELINE TRACE — route_message -> extract_state -> Q-table
       lookup -> apply_action_effect -> explain_decision, with the actual
       output of each stage printed so you can inspect it directly
    6. (optional) Live backend HTTP check, if `python ui/app.py` is running

Usage:
    python check_system.py
"""

import os
import sys
import json
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92m[PASS]\033[0m" if sys.stdout.isatty() else "[PASS]"
FAIL = "\033[91m[FAIL]\033[0m" if sys.stdout.isatty() else "[FAIL]"
WARN = "\033[93m[WARN]\033[0m" if sys.stdout.isatty() else "[WARN]"
INFO = "\033[94m[INFO]\033[0m" if sys.stdout.isatty() else "[INFO]"


def header(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# =============================================================================
# 0. Environment + packages
# =============================================================================
def check_environment():
    header("0. PYTHON ENVIRONMENT & PACKAGES")
    print(f"  Python: {sys.executable}")
    print(f"  Version: {sys.version.split()[0]}\n")

    required = ["fastapi", "uvicorn", "pydantic", "groq", "requests",
                "numpy", "gymnasium", "sklearn", "pandas"]
    all_ok = True
    for pkg in required:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {PASS} {pkg:<12} {version}")
        except ImportError:
            print(f"  {FAIL} {pkg:<12} NOT INSTALLED")
            all_ok = False
    return all_ok


# =============================================================================
# 1. LLM backend config
# =============================================================================
def check_llm_config():
    header("1. LLM BACKEND CONFIGURATION")
    backend = os.environ.get("LLM_BACKEND", "groq")
    print(f"  LLM_BACKEND = '{backend}'")

    if backend == "ollama":
        print(f"  {INFO} Using local Ollama — make sure `ollama serve` is running.")
        return True

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print(f"  {FAIL} GROQ_API_KEY is not set.")
        print(f"        PowerShell: $env:GROQ_API_KEY = \"your-key\"")
        return False

    masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
    print(f"  {PASS} GROQ_API_KEY is set ({masked})")

    # Actually try a live call — this is the only way to know the key is
    # valid, not just present.
    try:
        from groq import Groq
        client = Groq(api_key=key)
        r = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # llama-3.3-70b-versatile was deprecated by Groq
            messages=[{"role": "user", "content": "reply with just: ok"}],
            max_tokens=5,
        )
        reply = r.choices[0].message.content.strip()
        print(f"  {PASS} Live Groq call succeeded -> \"{reply}\"")
        return True
    except Exception as e:
        print(f"  {FAIL} Live Groq call failed: {e}")
        return False


# =============================================================================
# 2. Q-table database
# =============================================================================
def check_qtables():
    header("2. Q-TABLE DATABASE (qtables/qtables.db)")
    try:
        from qtable_store import list_qtables
    except ImportError as e:
        print(f"  {FAIL} Could not import qtable_store.py: {e}")
        return False

    try:
        entries = list_qtables()
    except Exception as e:
        print(f"  {FAIL} Could not read qtables.db: {e}")
        return False

    if not entries:
        print(f"  {WARN} No Q-tables found in qtables.db. Run training/train_*.py first.")
        return False

    stored_names = {e["name"] for e in entries}
    print(f"  Found {len(entries)} trained Q-table(s):\n")
    for e in sorted(entries, key=lambda x: x["name"]):
        print(f"    {PASS} {e['name']:<28} shape={e['shape']:<15} updated={e['updated_at'][:19]}")

    # Cross-check against the full expected set (15 tasks across 5 domains)
    expected = {
        "hospital_bed_allocation", "hospital_er_queue", "hospital_staff_allocation",
        "traffic_intersection", "traffic_pedestrian", "traffic_parking",
        "energy_solar_scheduling", "energy_battery_management", "energy_grid_interaction",
        "finance_trading", "finance_savings", "finance_budget",
        "agriculture_soil_preparation", "agriculture_irrigation", "agriculture_pest_control",
    }
    missing = expected - stored_names
    if missing:
        print(f"\n  {WARN} Missing Q-tables (not trained yet):")
        for m in sorted(missing):
            print(f"        - {m}")
    else:
        print(f"\n  {PASS} All 15 expected Q-tables are present.")

    return True


# =============================================================================
# 3. Knowledge base / RAG
# =============================================================================
def check_knowledge_base():
    header("3. KNOWLEDGE BASE (RAG policy retriever)")
    try:
        from policy_retriever import PolicyRetriever, build_query, TASK_SOURCE_MAP
    except ImportError as e:
        print(f"  {FAIL} Could not import policy_retriever.py: {e}")
        return False

    try:
        retriever = PolicyRetriever("knowledge_base")
    except FileNotFoundError as e:
        print(f"  {FAIL} {e}")
        return False

    print(f"  {PASS} Loaded {len(retriever.chunks)} policy chunks from knowledge_base/")

    query = build_query("er_queue", {"emergency_queue": 15, "normal_queue": 8}, "Serve Emergency")
    results = retriever.retrieve(query, top_k=1, source_filter=TASK_SOURCE_MAP["er_queue"])
    if results:
        print(f"  {PASS} Retrieval test OK — top match (score={results[0]['score']:.2f}):")
        print(f"        \"{results[0]['text'].splitlines()[0]}\"")
    else:
        print(f"  {WARN} Retrieval returned nothing for a known-good query.")
    return True


# =============================================================================
# 4. Live hospital state
# =============================================================================
def check_hospital_state():
    header("4. LIVE HOSPITAL STATE (hospital_state.json)")
    try:
        from state_manager import load_state
    except ImportError as e:
        print(f"  {FAIL} Could not import state_manager.py: {e}")
        return False

    try:
        state = load_state()
    except FileNotFoundError as e:
        print(f"  {FAIL} {e}")
        return False

    print(f"  {PASS} hospital_state.json loaded:")
    for task, s in state.items():
        clean = {k: v for k, v in s.items() if k != "last_updated"}
        print(f"        {task:<18} {clean}")
    return True


# =============================================================================
# 5. FULL PIPELINE TRACE (the real end-to-end test)
# =============================================================================
def check_full_pipeline():
    header("5. FULL PIPELINE TRACE — route -> extract -> decide -> explain")

    try:
        from llm_client import route_message, extract_state, explain_decision, TASK_FIELD_SPECS
        from policy_retriever import PolicyRetriever, build_query, TASK_SOURCE_MAP
        from qtable_store import load_qtable
        from agents.hospital_agent import discretize as hospital_discretize
        from environments.hospital_env import HospitalEnv
        import numpy as np
    except ImportError as e:
        print(f"  {FAIL} Import error: {e}")
        return False

    test_message = "There are 12 emergency patients and 6 normal patients waiting"
    print(f"  Test message: \"{test_message}\"\n")

    # ── Stage 1: Route ────────────────────────────────────────────────────
    print(f"  --- Stage 1: route_message() ---")
    try:
        tasks = route_message(test_message)
        print(f"  {PASS} Routed to: {tasks}")
        if "er_queue" not in tasks:
            print(f"  {WARN} Expected 'er_queue' in routed tasks, got {tasks}")
        task = "er_queue" if "er_queue" in tasks else (tasks[0] if tasks else None)
        if not task:
            print(f"  {FAIL} No task routed — cannot continue.")
            return False
    except Exception as e:
        print(f"  {FAIL} route_message() raised: {e}")
        traceback.print_exc()
        return False

    # ── Stage 2: Extract ──────────────────────────────────────────────────
    print(f"\n  --- Stage 2: extract_state() ---")
    try:
        known = {k: 0 for k in TASK_FIELD_SPECS[task]}
        extraction = extract_state(task, test_message, known)
        print(f"  {PASS} Extracted state: {extraction['state']}")
        print(f"        needs_clarification: {extraction['needs_clarification']}")
        print(f"        notes: {extraction['notes']}")
        if extraction["needs_clarification"]:
            print(f"  {WARN} Extractor asked for clarification instead of proceeding.")
            return False
        state = extraction["state"]
    except Exception as e:
        print(f"  {FAIL} extract_state() raised: {e}")
        traceback.print_exc()
        return False

    # ── Stage 3: Decide (direct Q-table lookup, same as ui/app.py) ──────────
    print(f"\n  --- Stage 3: Q-table decision ---")
    try:
        order = list(TASK_FIELD_SPECS[task].keys())
        obs = [state[k] for k in order]
        Q = load_qtable(f"hospital_{task}")
        s = hospital_discretize(obs, task)
        q_values = Q[s]
        actions = HospitalEnv(task=task).actions
        action_idx = int(np.argmax(q_values))
        action = actions[action_idx]
        print(f"  {PASS} Q-table shape: {Q.shape}")
        print(f"  {PASS} Discretized state index: {s}")
        print(f"  {PASS} Q-values: " + ", ".join(f"{a}={v:.3f}" for a, v in zip(actions, q_values)))
        print(f"  {PASS} Chosen action: {action}")
    except FileNotFoundError as e:
        print(f"  {FAIL} {e}")
        return False
    except Exception as e:
        print(f"  {FAIL} Decision step raised: {e}")
        traceback.print_exc()
        return False

    # ── Stage 4: Explain (RAG-grounded) ──────────────────────────────────────
    print(f"\n  --- Stage 4: explain_decision() ---")
    try:
        retriever = PolicyRetriever("knowledge_base")
        query = build_query(task, state, action)
        chunks = retriever.retrieve(query, top_k=2, source_filter=TASK_SOURCE_MAP[task])
        reason_hint = f"{state['emergency_queue']} emergency patients waiting — priority case"
        explanation = explain_decision(task, state, action, reason_hint, chunks)
        print(f"  {PASS} Explanation generated:")
        print(f"        \"{explanation}\"")
    except Exception as e:
        print(f"  {FAIL} explain_decision() raised: {e}")
        traceback.print_exc()
        return False

    print(f"\n  {PASS} FULL PIPELINE TRACE COMPLETE — every stage produced valid output.")
    return True


# =============================================================================
# 6. Live backend HTTP check (optional — only if server is running)
# =============================================================================
def check_live_backend():
    header("6. LIVE BACKEND (http://localhost:8000) — optional")
    try:
        import requests
    except ImportError:
        print(f"  {WARN} `requests` not installed, skipping this check.")
        return None

    try:
        r = requests.get("http://localhost:8000/api/health", timeout=3)
        if r.status_code == 200:
            print(f"  {PASS} Server is running and healthy.")
            return True
        else:
            print(f"  {WARN} Server responded with {r.status_code}.")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  {INFO} Server not running (this is fine — start it with `python ui/app.py` to test this layer too).")
        return None


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "#" * 70)
    print("#   UMORDA — FULL SYSTEM CHECK")
    print("#" * 70)

    results = {}
    results["environment"]    = check_environment()
    results["llm_config"]     = check_llm_config()
    results["qtables"]        = check_qtables()
    results["knowledge_base"] = check_knowledge_base()
    results["hospital_state"] = check_hospital_state()
    results["full_pipeline"]  = check_full_pipeline() if results["llm_config"] else False
    results["live_backend"]   = check_live_backend()

    header("SUMMARY")
    for name, ok in results.items():
        if ok is True:
            print(f"  {PASS} {name}")
        elif ok is False:
            print(f"  {FAIL} {name}")
        else:
            print(f"  {INFO} {name} (skipped)")

    critical = ["environment", "llm_config", "qtables", "full_pipeline"]
    if all(results.get(k) for k in critical):
        print(f"\n  {PASS} Core system is fully wired and working end-to-end.\n")
    else:
        print(f"\n  {WARN} One or more critical layers failed — see details above.\n")


if __name__ == "__main__":
    main()