"""
UMORDA — Groq LLM Integration (Hospital + Traffic)
File: llm_client.py

Responsibilities:
  1. route_message()      — reads user message, decides which task it belongs to
  2. extract_state()      — pulls numbers out of natural language into structured state
  3. explain_decision()   — explains the agent's decision in plain English (with policy)
  4. explain_ungrounded() — same but without policy context (for comparison)

Requires: pip install groq
Requires: GROQ_API_KEY environment variable set
"""

import os
import json
from groq import Groq

MODEL = "llama-3.3-70b-versatile"


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Windows: setx GROQ_API_KEY \"your-key-here\"  then restart terminal\n"
            "Get a free key at: https://console.groq.com"
        )
    return Groq(api_key=api_key)


def _call_llm(system_prompt, user_prompt, temperature=0.2, max_tokens=300):
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _parse_json(raw):
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"LLM did not return valid JSON.\nRaw output: {raw}")


# =============================================================================
# TASK FIELD SPECS
# Defines what numbers each task needs from the user's message.
# Adding a new domain = just add entries here. No other functions change.
# =============================================================================

TASK_FIELD_SPECS = {

    # ── HOSPITAL DOMAIN (Sourav's original — unchanged) ──────────────────────
    "bed_allocation": {
        "free_beds":        "current free hospital beds (integer, 0-20)",
        "waiting_patients": "patients currently waiting to be admitted (integer, 0-30)",
    },
    "er_queue": {
        "emergency_queue": "emergency patients waiting (integer, 0-10)",
        "normal_queue":    "normal/non-urgent patients waiting (integer, 0-20)",
    },
    "staff_allocation": {
        "available_doctors": "doctors currently on duty (integer, 1-15)",
        "patient_load":      "current patient load on staff (integer, 0-50)",
    },

    # ── TRAFFIC DOMAIN (Ador — v2 with wait time awareness) ──────────────────
    "intersection": {
        "cars_NS":       "total cars waiting North+South combined (integer, 0-9)",
        "cars_EW":       "total cars waiting East+West combined (integer, 0-9)",
        "current_phase": "which signal is green now: 0=GreenNS, 1=GreenEW (integer, 0-1)",
        "phase_elapsed": "how many steps current green signal has been active (integer, 0-9)",
        "wait_NS":       "how long North-South direction has been waiting (integer, 0-9)",
        "wait_EW":       "how long East-West direction has been waiting (integer, 0-9)",
    },
    "pedestrian": {
        "peds":      "number of pedestrians waiting to cross (integer, 0-9)",
        "vehs":      "number of vehicles waiting (integer, 0-9)",
        "ped_wait":  "how long pedestrians have been waiting (integer, 0-9)",
        "veh_wait":  "how long vehicles have been waiting (integer, 0-9)",
        "phase":     "current phase: 0=PedestrianPhase, 1=VehiclePhase (integer, 0-1)",
        "elapsed":   "steps current phase has been active (integer, 0-9)",
    },
    "parking": {
        "spots":      "available parking spots remaining (integer, 0-19)",
        "incoming":   "vehicles approaching the lot right now (integer, 0-9)",
        "queue_wait": "how long vehicles have been queuing to enter (integer, 0-9)",
        "occupancy":  "lot fullness: 0=<25%%, 1=25-50%%, 2=50-75%%, 3=75-100%%, 4=FULL (integer, 0-4)",
    },
}

TASK_DESCRIPTIONS = {
    # HOSPITAL
    "bed_allocation":   "hospital bed management — admitting or rejecting patients based on bed availability",
    "er_queue":         "emergency room triage — serving emergency vs normal queue patients",
    "staff_allocation": "doctor staffing — assigning more or fewer doctors based on patient load",
    # TRAFFIC
    "intersection": "traffic intersection signal control — switching green light between North-South and East-West based on car counts and wait times",
    "pedestrian":   "pedestrian crossing control — allowing pedestrians or vehicles to go, prioritizing safety",
    "parking":      "parking lot entry management — opening Zone A, Zone B, or closing entry based on available spots and queue",
}


# =============================================================================
# 1. ROUTER — which task does this message belong to?
# =============================================================================
def route_message(user_message, conversation_history=None):
    """
    Reads the user's message and decides which task(s) it belongs to.
    Returns a list of task names e.g. ['intersection', 'pedestrian']
    """
    task_desc = "\n".join(f"- {k}: {v}" for k, v in TASK_DESCRIPTIONS.items())

    history_text = ""
    if conversation_history:
        history_text = "\n\nRecent conversation:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation_history[-6:]
        )

    system_prompt = f"""You are a router for UMORDA — a system that manages
hospital resources AND traffic signals.

Read the user message and decide which task(s) it is about:

{task_desc}

Rules:
- A message can be about MULTIPLE tasks — include all that apply.
- If it is a follow-up, check conversation history and keep same task.
- If message is unrelated to any task, return empty list.
- Output ONLY this JSON: {{"tasks": ["task_name", ...]}}
"""
    user_prompt = f'User message: "{user_message}"{history_text}\n\nOutput JSON.'

    raw    = _call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=100)
    parsed = _parse_json(raw)
    tasks  = parsed.get("tasks", [])
    return [t for t in tasks if t in TASK_FIELD_SPECS]


# =============================================================================
# 2. EXTRACTOR — pull numbers out of natural language
# =============================================================================
def extract_state(task, user_message, known_state, conversation_history=None):
    """
    Turns the user's natural language into exact numbers the Q-table needs.

    Returns a dict with:
      state                  — updated numeric state
      needs_clarification    — True if message is too vague or impossible
      clarification_question — what to ask the user if unclear
      notes                  — what was inferred vs directly stated
    """
    fields     = TASK_FIELD_SPECS[task]
    field_desc = "\n".join(f"- {k}: {v}" for k, v in fields.items())

    history_text = ""
    if conversation_history:
        history_text = "\n\nConversation history:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation_history[-6:]
        )

    system_prompt = f"""You are a data extractor for UMORDA.
Task: {task} — {TASK_DESCRIPTIONS.get(task, '')}

Extract these fields from the user message:
{field_desc}

Rules:
- Start from the known current values.
- Update fields the message mentions. Keep others unchanged.
- Vague words → reasonable numbers ("a lot" → high, "a few" → 2-4).
- Follow-ups like "a few more arrived" → ADD to known value.
- Impossible values (e.g. 5000 cars) → needs_clarification=true + question.
- Completely unrelated message → needs_clarification=true.
- Output ONLY this JSON:
{{
  "state": {{...fields...}},
  "needs_clarification": true/false,
  "clarification_question": "..." or null,
  "notes": "short note on what was inferred"
}}
"""
    user_prompt = f"""Current known state: {json.dumps(known_state)}{history_text}

User message: "{user_message}"

Output JSON."""

    raw    = _call_llm(system_prompt, user_prompt, temperature=0.2, max_tokens=350)
    parsed = _parse_json(raw)

    result_state = dict(known_state)
    for key in fields.keys():
        if key in parsed.get("state", {}):
            try:
                result_state[key] = int(parsed["state"][key])
            except (TypeError, ValueError):
                pass

    return {
        "state":                  result_state,
        "needs_clarification":    bool(parsed.get("needs_clarification", False)),
        "clarification_question": parsed.get("clarification_question"),
        "notes":                  parsed.get("notes", ""),
    }


# =============================================================================
# 3. EXPLAINER — turn agent decision into plain English (with policy)
# =============================================================================
def explain_decision(task, state, action, reason_hint, policy_chunks):
    """Explains the agent's decision using policy context (RAG-grounded)."""
    policy_text = "\n\n".join(c["text"] for c in policy_chunks)

    system_prompt = """You are an assistant explaining decisions made by the
UMORDA reinforcement learning agent. Write a clear 2-4 sentence explanation.
Use the policy text naturally. Do not invent facts. Be practical and direct."""

    user_prompt = f"""Task: {task}
Situation: {json.dumps(state)}
Decision: {action}
Reason: {reason_hint}

Policy context:
{policy_text}

Explain this decision in plain language."""

    return _call_llm(system_prompt, user_prompt, temperature=0.4, max_tokens=250)


# =============================================================================
# 4. EXPLAINER — without policy (for comparison)
# =============================================================================
def explain_ungrounded(task, state, action, reason_hint):
    """Same as explain_decision but without policy — for comparison."""
    system_prompt = """You are an assistant explaining decisions made by the
UMORDA reinforcement learning agent. Write a clear 2-4 sentence explanation
using your general knowledge. Be practical and direct."""

    user_prompt = f"""Task: {task}
Situation: {json.dumps(state)}
Decision: {action}
Reason: {reason_hint}

Explain this decision in plain language."""

    return _call_llm(system_prompt, user_prompt, temperature=0.4, max_tokens=250)

# =============================================================================
# ENERGY DOMAIN SPECS (added below existing hospital + traffic specs)
# =============================================================================
ENERGY_TASK_FIELD_SPECS = {
    "solar_scheduling": {
        "solar_output":     "how much solar power the balcony panels are generating (0-9, 0=none, 9=maximum)",
        "home_consumption": "current home electricity usage (0-9)",
        "battery_level":    "current battery charge level (0-9, 0=empty, 9=full)",
        "time_of_day":      "time of day: 0=Morning, 1=Afternoon, 2=Evening, 3=Night (integer, 0-3)",
    },
    "battery_management": {
        "battery_level":    "current battery charge (0-9, 0=empty, 9=full)",
        "solar_output":     "current solar generation (0-9)",
        "grid_price":       "electricity price: 0=Cheap, 1=Normal, 2=Expensive (integer, 0-2)",
        "home_consumption": "current home electricity usage (0-9)",
    },
    "grid_interaction": {
        "grid_price":       "electricity price: 0=Cheap, 1=Normal, 2=Expensive (integer, 0-2)",
        "solar_surplus":    "extra solar power beyond home needs (0-9)",
        "battery_level":    "current battery charge (0-9)",
        "home_consumption": "current home electricity usage (0-9)",
    },
}

ENERGY_TASK_DESCRIPTIONS = {
    "solar_scheduling": "balcony solar panel power scheduling — deciding whether to use solar directly, store it in battery, or buy from grid",
    "battery_management": "battery storage optimization — deciding when to charge, discharge, or keep the battery idle based on solar availability and grid price",
    "grid_interaction": "grid energy exchange — deciding when to buy electricity from the grid, sell surplus solar to the grid, or stay self-sufficient",
}

# Merge energy specs into main dicts so existing functions work automatically
TASK_FIELD_SPECS.update(ENERGY_TASK_FIELD_SPECS)
TASK_DESCRIPTIONS.update(ENERGY_TASK_DESCRIPTIONS)
