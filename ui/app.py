"""
UMORDA — FastAPI Backend for the Web UI
File: ui/app.py

Serves the same 3-layer pipeline that main.py / hospital_assistant.py /
*_assistant.py already expose on the CLI, over HTTP, for
ui/index.html + ui/script.js:

    1. route_message()    -> which task is this about?
    2. extract_state()    -> pull numeric state out of the sentence
    3. trained Q-table    -> optimal action (+ Q-values for the UI's bars)
    4. explain_decision() -> RAG-grounded plain-English explanation

Endpoints (consumed by ui/script.js):
    GET  /api/health    -> {"status": "ok"}
    GET  /api/domains   -> {domain_key: {"label": ...}, ...}
    GET  /api/history   -> [{"query":..., "domain":..., "action":..., "confidence":...}, ...]
    POST /api/query     -> {"route": {...}, "extract": {...}, "decide": {...}, "explain": "..."}

Also serves the static frontend itself (index.html / style.css / script.js)
at "/", so `python ui/app.py` is enough to try the whole thing end-to-end —
no separate static file server needed.

Setup:
    pip install fastapi "uvicorn[standard]" pydantic
    export GROQ_API_KEY="your-key"        # or: export LLM_BACKEND=ollama
    python ui/app.py

Then open:
    http://localhost:8000
"""

import os
import sys
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Make the repo root importable (this file lives in ui/) ───────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from environments.hospital_env import HospitalEnv
from environments.traffic_env import TrafficEnv
from environments.energy_env import EnergyEnv
from agents.hospital_agent import discretize as hospital_discretize

from llm_client import (
    route_message, extract_state, explain_decision, explain_ungrounded,
    TASK_FIELD_SPECS,
)
from policy_retriever import PolicyRetriever, build_query, TASK_SOURCE_MAP

# All trained Q-tables live in qtables/qtables.db — same single source of
# truth main.py uses. Do not fall back to standalone .npy files.
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

try:
    from hospital_network import (
        seed_demo_network, update_this_hospital_status,
        suggest_nearby_hospital, format_suggestion,
    )
    HAS_HOSPITAL_NETWORK = True
except ImportError:
    HAS_HOSPITAL_NETWORK = False

try:
    from state_manager import get_task_state, apply_action_effect, merge_new_arrivals
    HAS_STATE_MANAGER = True
except (ImportError, FileNotFoundError):
    HAS_STATE_MANAGER = False

HOSPITAL_TASKS = {"bed_allocation", "er_queue", "staff_allocation"}

# Matches er_queue_policy.md's guidance that >5 waiting emergency patients
# is a "resource strain event" — used to trigger a network suggestion.
ER_QUEUE_STRAIN_THRESHOLD = 10


# =============================================================================
# Domain / task registry — mirrors main.py's TASK_DOMAIN + DEFAULT_STATE
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
    "trading":             "finance",
    "savings":             "finance",
    "budget":              "finance",
    "soil_preparation":   "agriculture",
    "irrigation":         "agriculture",
    "pest_control":       "agriculture",
}

DOMAIN_LABELS = {
    "hospital":    "Hospital",
    "traffic":     "Traffic",
    "energy":      "Energy",
    "finance":     "Finance",
    "agriculture": "Agriculture",
}

TASK_LABELS = {t: t.replace("_", " ").title() for t in TASK_DOMAIN}

DEFAULT_STATE = {
    "bed_allocation":   {"free_beds": 8, "waiting_patients": 5},
    "er_queue":         {"emergency_queue": 0, "normal_queue": 3},
    "staff_allocation": {"available_doctors": 6, "patient_load": 15},
    "intersection": {"cars_NS": 2, "cars_EW": 2, "current_phase": 0,
                      "phase_elapsed": 0, "wait_NS": 0, "wait_EW": 0},
    "pedestrian":   {"peds": 0, "vehs": 0, "ped_wait": 0,
                      "veh_wait": 0, "phase": 1, "elapsed": 0},
    "parking":      {"spots": 10, "incoming": 0, "queue_wait": 0, "occupancy": 0},
    "solar_scheduling":   {"solar_output": 0, "home_consumption": 3,
                            "battery_level": 5, "time_of_day": 1},
    "battery_management": {"battery_level": 5, "solar_output": 0,
                            "grid_price": 1, "home_consumption": 3},
    "grid_interaction":   {"grid_price": 1, "solar_surplus": 0,
                            "battery_level": 5, "home_consumption": 3},
    "trading": {"price_trend": 0, "shares_held": 0, "cash": 1000, "portfolio_value": 1000},
    "savings": {"monthly_income": 50, "current_savings": 200,
                "expenses": 40, "months_remaining": 12},
    "budget":  {"total_budget": 800, "amount_spent": 0,
                "urgent_requests": 0, "departments_remaining": 5},
    "soil_preparation": {"soil_ph": 6, "organic_matter": 30,
                          "drainage_quality": 30, "days_remaining": 25},
    "irrigation":        {"water_reservoir": 60, "crop_stress": 20,
                           "rainfall_trend": 0, "days_remaining": 40},
    "pest_control":       {"total_resource": 750, "resource_used": 0,
                            "urgent_outbreaks": 0, "plots_remaining": 6},
}


def _state_order(task):
    return list(TASK_FIELD_SPECS[task].keys())


# =============================================================================
# Per-domain decision + Q-value extraction
# Every function returns (action_name, [(action_name, q_value), ...], error)
# exactly one of (action_name, error) is None.
# =============================================================================
def _decide_hospital(task, state):
    order = _state_order(task)
    obs   = [state[k] for k in order]
    try:
        Q = db_load_qtable(f"hospital_{task}")
    except FileNotFoundError as e:
        return None, None, str(e)
    try:
        s = hospital_discretize(obs, task)
        q = Q[s]
    except (IndexError, KeyError) as e:
        return None, None, f"Discretization/Q-table mismatch for hospital/{task} ({e})."
    actions = HospitalEnv(task=task).actions
    action  = actions[int(np.argmax(q))]
    return action, list(zip(actions, q.tolist())), None


def _decide_traffic(task, state):
    env   = TrafficEnv(task=task)
    order = _state_order(task)
    raw   = [state[k] for k in order]
    encoded        = env._encode_state(raw)
    env._raw_state = raw
    try:
        Q = db_load_qtable(f"traffic_{task}")
    except FileNotFoundError as e:
        return None, None, str(e)
    if encoded >= Q.shape[0]:
        return None, None, "State index out of range for saved Q-table (shape mismatch)."
    q            = Q[encoded]
    q_action     = int(np.argmax(q))
    final_action = env._safety_override(q_action)   # hard safety guarantee
    actions      = env.cfg["action_meanings"]
    return actions[final_action], list(zip(actions, q.tolist())), None


def _decide_energy(task, state):
    env   = EnergyEnv(task=task)
    order = _state_order(task)
    raw   = [state[k] for k in order]
    encoded = env._encode_state(raw)
    try:
        Q = db_load_qtable(f"energy_{task}")
    except FileNotFoundError as e:
        return None, None, str(e)
    if encoded >= Q.shape[0]:
        return None, None, "State index out of range for saved Q-table (shape mismatch)."
    q       = Q[encoded]
    actions = env.cfg["action_meanings"]
    return actions[int(np.argmax(q))], list(zip(actions, q.tolist())), None


def _decide_finance(task, state):
    if not HAS_FINANCE_AGENT:
        return None, None, "agents/finance_agent.py not found."
    try:
        agent = FinanceAgent(task=task)
    except FileNotFoundError as e:
        return None, None, str(e)
    _, q_values, action_name = agent.get_action(state)
    return action_name, list(zip(agent.actions, np.asarray(q_values).tolist())), None


def _decide_agriculture(task, state):
    if not HAS_AGRICULTURE_AGENT:
        return None, None, "agents/agriculture_agent.py failed to import."
    order = _state_order(task)
    obs   = [state[k] for k in order]
    try:
        Q = db_load_qtable(f"agriculture_{task}")
    except FileNotFoundError as e:
        return None, None, str(e)
    try:
        s = agriculture_discretize(obs, task)
        q = Q[s]
    except (IndexError, KeyError) as e:
        return None, None, f"Discretization/Q-table mismatch for agriculture/{task} ({e})."
    actions = AgricultureEnv(task=task).actions
    return actions[int(np.argmax(q))], list(zip(actions, q.tolist())), None


DECIDERS = {
    "hospital":    _decide_hospital,
    "traffic":     _decide_traffic,
    "energy":      _decide_energy,
    "finance":     _decide_finance,
    "agriculture": _decide_agriculture,
}


def get_decision_with_qvalues(task, state):
    domain = TASK_DOMAIN.get(task)
    if domain not in DECIDERS:
        return None, None, f"Unknown domain for task '{task}'."
    return DECIDERS[domain](task, state)


def get_reason_hint(task, state, action):
    """
    Builds a SPECIFIC, numbers-grounded reason for explain_decision() to work
    from, instead of the generic "selected as optimal by Q-learning" boilerplate.
    This is what actually makes the LLM's explanation concrete (mirrors
    hospital_assistant.py's get_reason_hint, extended to every domain).
    """
    if task == "bed_allocation":
        beds = state["free_beds"]
        if action == "Admit":
            return f"{beds} beds available — sufficient capacity to admit safely"
        elif action == "Transfer":
            return f"only {beds} beds available — preserving capacity by transferring"
        else:
            return f"{beds} beds available and transfer capacity exhausted — rejection is a last resort"

    elif task == "er_queue":
        eq, nq = state["emergency_queue"], state["normal_queue"]
        if action == "Serve Emergency":
            return f"{eq} emergency patients waiting (normal queue: {nq}) — emergency always takes strict priority while non-empty"
        else:
            return f"emergency queue is empty — safe to serve normal queue ({nq} waiting)"

    elif task == "staff_allocation":
        load, docs = state["patient_load"], state["available_doctors"]
        if action == "Assign More Staff":
            return f"patient load is {load} with {docs} doctors on duty — above safe staffing threshold"
        elif action == "Reduce Staff":
            return f"patient load is {load} with {docs} doctors on duty — low enough to reduce cost safely"
        else:
            return f"patient load is {load} with {docs} doctors on duty — within balanced staffing range"

    elif task == "intersection":
        return (f"NS: {state['cars_NS']} cars waited {state['wait_NS']} steps, "
                f"EW: {state['cars_EW']} cars waited {state['wait_EW']} steps")

    elif task == "pedestrian":
        return (f"{state['peds']} pedestrians waited {state['ped_wait']} steps, "
                f"{state['vehs']} vehicles waited {state['veh_wait']} steps")

    elif task == "parking":
        return (f"{state['spots']} spots available, {state['incoming']} incoming, "
                f"queue waited {state['queue_wait']} steps")

    elif task == "solar_scheduling":
        return (f"solar={state['solar_output']}/9, home use={state['home_consumption']}/9, "
                f"battery={state['battery_level']}/9")

    elif task == "battery_management":
        return (f"battery={state['battery_level']}/9, solar={state['solar_output']}/9, "
                f"grid price level={state['grid_price']}")

    elif task == "grid_interaction":
        return (f"grid price level={state['grid_price']}, surplus={state['solar_surplus']}/9, "
                f"battery={state['battery_level']}/9")

    elif task == "trading":
        return f"price trend={state['price_trend']}, shares held={state['shares_held']}, cash=${state['cash']:.0f}"

    elif task == "savings":
        return (f"income=${state['monthly_income']:.0f}, savings=${state['current_savings']:.0f}, "
                f"expenses=${state['expenses']:.0f}, {state['months_remaining']:.0f} months left")

    elif task == "budget":
        remaining = state["total_budget"] - state["amount_spent"]
        return (f"${remaining:.0f} of ${state['total_budget']:.0f} remaining, "
                f"{state['urgent_requests']:.0f} urgent requests pending")

    elif task == "soil_preparation":
        return (f"pH={state['soil_ph']}, organic matter={state['organic_matter']}%, "
                f"drainage={state['drainage_quality']}%, {state['days_remaining']} days left")

    elif task == "irrigation":
        return (f"reservoir={state['water_reservoir']}, crop stress={state['crop_stress']}, "
                f"rainfall trend={state['rainfall_trend']}")

    elif task == "pest_control":
        remaining = state["total_resource"] - state["resource_used"]
        return (f"{remaining:.0f} resource remaining, {state['urgent_outbreaks']:.0f} urgent outbreaks, "
                f"{state['plots_remaining']:.0f} plots left")

    return f"Selected as the optimal action by the trained Q-learning policy for '{task}'."


def _hospital_network_suggestion(task, state, action):
    """
    If this hospital can't help (out of beds, or ER overloaded), find the
    nearest connected hospital that currently has capacity. Mirrors the
    trigger logic in hospital_assistant.py's process_task().
    Returns a human-readable line, or None if not applicable / unavailable.
    """
    if not HAS_HOSPITAL_NETWORK:
        return None

    if task == "bed_allocation":
        update_this_hospital_status(free_beds=state["free_beds"], free_er_beds=0)
        if action in ("Transfer", "Reject"):
            suggestion = suggest_nearby_hospital(need_er=False)
            return format_suggestion(suggestion, need_er=False,
                                      reason=f"Only {state['free_beds']} beds available here")

    elif task == "er_queue":
        eq = state.get("emergency_queue", 0)
        if eq >= ER_QUEUE_STRAIN_THRESHOLD:
            suggestion = suggest_nearby_hospital(need_er=True)
            return format_suggestion(suggestion, need_er=True,
                                      reason=f"ER overloaded here ({eq} emergency patients waiting)")

    return None


def _state_diff_summary(task, before, after):
    """
    Deterministic, non-LLM-generated account of exactly what changed on this
    turn. Guaranteed accurate (unlike the LLM's prose explanation), since it's
    computed directly from the before/after dicts rather than generated text.
    Returns None if there's nothing to report (e.g. task not stateful, or
    state_manager.py unavailable).
    """
    if before is after or not isinstance(before, dict) or not isinstance(after, dict):
        return None
    changes = []
    for key in before:
        if key == "last_updated":
            continue
        b, a = before.get(key), after.get(key)
        if b != a:
            changes.append(f"{key.replace('_', ' ')}: {b} → {a}")
    if not changes:
        return None
    return "📊 Live update — " + "; ".join(changes) + "."


def _confidence_from_qvalues(q_pairs, chosen_action):
    """
    Softmax probability of the chosen action, as a 0..1 'confidence' score.
    (The UI multiplies this by 100 for a percentage / progress bar.)
    """
    values  = np.array([v for _, v in q_pairs], dtype=float)
    shifted = values - values.max()
    exp     = np.exp(shifted)
    probs   = exp / exp.sum()
    idx     = [name for name, _ in q_pairs].index(chosen_action)
    return float(probs[idx])


# =============================================================================
# Explanation (RAG-grounded, falls back to ungrounded if retrieval is empty)
# — matches main.py's get_explanation()
# =============================================================================
def get_explanation(retriever, task, state, action):
    reason_hint = get_reason_hint(task, state, action)
    chunks = []
    if task in TASK_SOURCE_MAP:
        query  = build_query(task, state, action)
        chunks = retriever.retrieve(query, top_k=2, source_filter=TASK_SOURCE_MAP[task])
    if chunks:
        return explain_decision(task, state, action, reason_hint, chunks)
    return explain_ungrounded(task, state, action, reason_hint)


# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(title="UMORDA Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared "session" of live state + conversation memory, matching the
# single-user CLI demos (hospital_assistant.py, main.py, etc). If this needs
# multiple concurrent users later, key these three by a client/session id
# passed from the frontend instead of using module-level globals.
known_state = {t: dict(DEFAULT_STATE[t]) for t in DEFAULT_STATE}

# For hospital tasks specifically, hydrate from the real hospital_state.json
# on disk (via state_manager.py) instead of hardcoded defaults, so bed/queue/
# staff counts survive server restarts and reflect whatever was last written.
if HAS_STATE_MANAGER:
    for _task in HOSPITAL_TASKS:
        try:
            known_state[_task] = get_task_state(_task)
        except (FileNotFoundError, ValueError):
            pass  # fall back to the hardcoded default already set above

conversation_history  = []
query_history         = []   # newest first, for GET /api/history
MAX_CONVO_HISTORY     = 12
MAX_QUERY_HISTORY     = 100

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever(os.path.join(REPO_ROOT, "knowledge_base"))
    return _retriever


# Seed the (demo/mock) hospital network so bed_allocation Transfer/Reject and
# er_queue overload decisions can suggest a nearby hospital with capacity —
# same as hospital_assistant.py does on the CLI. Safe no-op if
# hospital_network.py isn't importable.
if HAS_HOSPITAL_NETWORK:
    seed_demo_network(this_hospital_free_beds=known_state["bed_allocation"]["free_beds"])


class QueryRequest(BaseModel):
    query: str
    domain: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/domains")
def domains():
    return {domain: {"label": label} for domain, label in DOMAIN_LABELS.items()}


@app.get("/api/history")
def history():
    return query_history[:30]


@app.post("/api/query")
def query(req: QueryRequest):
    user_message = req.query.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty query.")

    retriever = get_retriever()

    conversation_history.append({"role": "user", "content": user_message})
    del conversation_history[:-MAX_CONVO_HISTORY]

    # ── 1. Route ───────────────────────────────────────────────────────────
    tasks = route_message(user_message, conversation_history)
    if req.domain:
        # Respect an explicit domain filter from the UI's dropdown. If the
        # router didn't surface a task in that domain, fall back to any task
        # belonging to it rather than refusing outright.
        in_domain = [t for t in tasks if TASK_DOMAIN.get(t) == req.domain]
        tasks = in_domain or [t for t in TASK_DOMAIN if TASK_DOMAIN[t] == req.domain]

    if not tasks:
        raise HTTPException(
            status_code=422,
            detail=("Couldn't match this to a known task. Try mentioning beds/ER/staffing, "
                     "an intersection/crossing/parking lot, solar/battery/grid, "
                     "trading/savings/budget, or soil/irrigation/pests specifically."),
        )

    task   = tasks[0]
    domain = TASK_DOMAIN[task]

    # ── 2. Extract ─────────────────────────────────────────────────────────
    extraction = extract_state(task, user_message, known_state[task], conversation_history)
    if extraction["needs_clarification"]:
        question = extraction["clarification_question"] or "Could you clarify the numbers involved?"
        raise HTTPException(status_code=422, detail=question)

    known_state[task] = extraction["state"]
    state = known_state[task]   # decision-time state — used for the decision, reason hint, and explanation

    # Persist user-stated numbers to hospital_state.json for hospital tasks,
    # so the live bed/queue/staff picture survives restarts and other clients
    # (CLI assistants) see the same numbers.
    if HAS_STATE_MANAGER and task in HOSPITAL_TASKS:
        merge_new_arrivals(task, state)

    # ── 3. Decide ──────────────────────────────────────────────────────────
    action, q_pairs, error = get_decision_with_qvalues(task, state)
    if error:
        raise HTTPException(status_code=500, detail=error)

    confidence = _confidence_from_qvalues(q_pairs, action)
    q_sorted = sorted(
        [{"action": name, "q_value": round(float(v), 4)} for name, v in q_pairs],
        key=lambda x: x["q_value"], reverse=True,
    )

    # ── 4. Explain ─────────────────────────────────────────────────────────
    explanation = get_explanation(retriever, task, state, action)

    # ── 4b. Hospital Suggestion Network ──────────────────────────────────────
    # If this hospital can't help (out of beds, ER overloaded), surface the
    # nearest connected hospital with capacity, right alongside the explanation.
    network_line = _hospital_network_suggestion(task, state, action)
    if network_line:
        explanation = f"{explanation}\n\n🏥 {network_line}"

    # ── 4c. Apply the action's real effect for NEXT time ─────────────────────
    # Admit -> free_beds -= 1, Serve Emergency -> emergency_queue -= 1, etc.
    # Written to hospital_state.json now that the response for THIS turn is
    # already built from the pre-action numbers above.
    state_after_action = state
    diff_summary = None
    if HAS_STATE_MANAGER and task in HOSPITAL_TASKS:
        state_after_action = apply_action_effect(task, action)
        known_state[task] = state_after_action
        diff_summary = _state_diff_summary(task, state, state_after_action)
        if diff_summary:
            explanation = f"{explanation}\n\n{diff_summary}"

    conversation_history.append({"role": "assistant", "content": f"Processed: {task} -> {action}"})

    query_history.insert(0, {
        "id":         str(uuid.uuid4()),
        "query":      user_message,
        "domain":     domain,
        "task":       task,
        "action":     action,
        "confidence": round(confidence, 4),
        "timestamp":  time.time(),
    })
    del query_history[MAX_QUERY_HISTORY:]

    return {
        "route": {
            "domain":       domain,
            "domain_label": DOMAIN_LABELS[domain],
            "task":         task,
            "task_label":   TASK_LABELS[task],
        },
        "extract": {
            "task":  task,
            "state": state,
            "state_after_action": state_after_action,
            "diff_summary": diff_summary,
            "notes": extraction["notes"],
        },
        "decide": {
            "chosen_action": action,
            "confidence":    confidence,
            "q_values":      q_sorted,
        },
        "network_suggestion": network_line,
        "explain": explanation,
    }


# ── Serve the static frontend (index.html / style.css / script.js) at "/" ────
# Mounted last so the /api/* routes above are matched first.
app.mount("/", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__)), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("GROQ_API_KEY") and os.environ.get("LLM_BACKEND", "groq") != "ollama":
        print("\n  WARNING: GROQ_API_KEY not set (or set LLM_BACKEND=ollama).")
        print("  The route_message/extract_state/explain_decision calls will fail without it.\n")

    print("\n  UMORDA backend starting on http://localhost:8000")
    print("  Frontend served from the same URL — API lives under /api/*.\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)