"""
UMORDA — Groq LLM Integration (Hospital + Traffic + Energy)
File: llm_client.py

FIXED VERSION:
- Smarter router (only activates tasks user actually mentioned)
- Better extractor (understands scale properly)
- Better explainer (no raw Q-values, human friendly language)
- Scale context added so LLM never confuses price/battery levels
"""

import os, json
import requests
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

# ── Backend switch ───────────────────────────────────────────────────────
# Set LLM_BACKEND=ollama in your environment to run everything (router,
# extractor, explainer) on your local 7B model instead of Groq.
# Windows PowerShell:   $env:LLM_BACKEND="ollama"
# Windows CMD:          set LLM_BACKEND=ollama
# Mac/Linux:            export LLM_BACKEND=ollama
LLM_BACKEND  = os.environ.get("LLM_BACKEND", "groq")   # "groq" or "ollama"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2:7b")
OLLAMA_URL   = "http://localhost:11434/api/chat"

# Scale reference — used in extractor and explainer prompts
SCALE_CONTEXT = """
IMPORTANT SCALE REFERENCE (always use this when interpreting numbers):
- solar_output    : 0=No solar at all, 1-3=Low/weak, 4-6=Moderate, 7-9=Strong/maximum
- home_consumption: 0-2=Very low usage, 3-5=Normal usage, 6-9=High usage
- battery_level   : 0-1=Nearly empty, 2-4=Low, 5-6=Medium, 7-8=High, 9=Full
- grid_price      : 0=CHEAP (good time to buy), 1=NORMAL, 2=EXPENSIVE (avoid buying!)
- solar_surplus   : 0=No surplus, 1-3=Small surplus, 4-6=Good surplus, 7-9=Large surplus
- solar_scheduling time_of_day: 0=Morning, 1=Afternoon(peak sun), 2=Evening, 3=Night(no sun)
"""


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Windows: setx GROQ_API_KEY \"your-key-here\" then restart terminal\n"
            "Get a free key at: https://console.groq.com"
        )
    return Groq(api_key=api_key)


def _call_ollama(system_prompt, user_prompt, temperature=0.2, max_tokens=400):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def _call_llm(system_prompt, user_prompt, temperature=0.2, max_tokens=400):
    if LLM_BACKEND == "ollama":
        return _call_ollama(system_prompt, user_prompt, temperature, max_tokens)

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
        raise ValueError(f"LLM did not return valid JSON.\nRaw: {raw}")


# =============================================================================
# TASK FIELD SPECS
# =============================================================================
TASK_FIELD_SPECS = {

    # HOSPITAL (Sourav — unchanged)
    "bed_allocation": {
        "free_beds":        "current free hospital beds (integer, 0-20)",
        "waiting_patients": "patients waiting to be admitted (integer, 0-30)",
    },
    "er_queue": {
        "emergency_queue": "emergency patients waiting (integer, 0-10)",
        "normal_queue":    "normal patients waiting (integer, 0-20)",
    },
    "staff_allocation": {
        "available_doctors": "doctors on duty (integer, 1-15)",
        "patient_load":      "current patient load (integer, 0-50)",
    },

    # TRAFFIC (Ador)
    "intersection": {
        "cars_NS":       "total cars North+South (integer, 0-9)",
        "cars_EW":       "total cars East+West (integer, 0-9)",
        "current_phase": "current signal: 0=GreenNS, 1=GreenEW (integer, 0-1)",
        "phase_elapsed": "steps current signal active (integer, 0-9)",
        "wait_NS":       "how long NS has been waiting (integer, 0-9)",
        "wait_EW":       "how long EW has been waiting (integer, 0-9)",
    },
    "pedestrian": {
        "peds":     "pedestrians waiting (integer, 0-9)",
        "vehs":     "vehicles waiting (integer, 0-9)",
        "ped_wait": "how long peds waited (integer, 0-9)",
        "veh_wait": "how long vehs waited (integer, 0-9)",
        "phase":    "0=PedPhase, 1=VehiclePhase (integer, 0-1)",
        "elapsed":  "steps in current phase (integer, 0-9)",
    },
    "parking": {
        "spots":      "available spots (integer, 0-19)",
        "incoming":   "vehicles incoming (integer, 0-9)",
        "queue_wait": "how long queue waited (integer, 0-9)",
        "occupancy":  "0=<25%%, 1=25-50%%, 2=50-75%%, 3=75-100%%, 4=FULL (integer, 0-4)",
    },

    # ENERGY (Ador)
    "solar_scheduling": {
        "solar_output":     "solar power being generated RIGHT NOW (integer, 0-9). IMPORTANT: rain/clouds/night = 0, weak sun = 1-3, good sun = 5-7, bright afternoon = 8-9",
        "home_consumption": "current home electricity usage (integer, 0-9). normal home = 3-4",
        "battery_level":    "current battery charge (integer, 0-9). 0=empty, 9=full, 5=medium",
        "time_of_day":      "0=Morning, 1=Afternoon, 2=Evening, 3=Night (integer, 0-3)",
    },
    "battery_management": {
        "battery_level":    "current battery charge (integer, 0-9). 0=empty, 9=full, 5=medium",
        "solar_output":     "solar being generated now (integer, 0-9). rain/night = 0",
        "grid_price":       "MUST BE: 0=Cheap, 1=Normal, 2=Expensive. High/expensive price = 2",
        "home_consumption": "current home usage (integer, 0-9)",
    },
    "grid_interaction": {
        "grid_price":       "MUST BE: 0=Cheap, 1=Normal, 2=Expensive. High/expensive price = 2",
        "solar_surplus":    "extra solar beyond home needs (integer, 0-9). rain/night = 0",
        "battery_level":    "current battery charge (integer, 0-9). 0=empty, 9=full, 5=medium",
        "home_consumption": "current home usage (integer, 0-9)",
    },
}

TASK_DESCRIPTIONS = {
    # HOSPITAL
    "bed_allocation":   "hospital bed management",
    "er_queue":         "emergency room triage",
    "staff_allocation": "doctor staffing levels",
    # TRAFFIC
    "intersection": "traffic intersection signal control",
    "pedestrian":   "pedestrian crossing control",
    "parking":      "parking lot entry management",
    # ENERGY
    "solar_scheduling":   "deciding how to use solar power being generated right now (use directly, store in battery, or buy from grid)",
    "battery_management": "deciding when to charge, discharge, or keep battery idle",
    "grid_interaction":   "deciding when to buy from grid, sell surplus solar to grid, or stay self-sufficient",
}

# Keywords that strongly suggest each energy task
ENERGY_TASK_KEYWORDS = {
    "solar_scheduling": ["solar", "sun", "panel", "generating", "rain", "cloud", "shine", "light", "dark"],
    "battery_management": ["battery", "charge", "discharge", "store", "stored", "batter"],
    "grid_interaction": ["grid", "sell", "buy", "surplus", "export", "import", "self-sufficient", "independent"],
}


# =============================================================================
# 1. ROUTER — FIXED: only routes to tasks user actually mentioned
# =============================================================================
def route_message(user_message, conversation_history=None):
    """
    Routes user message to relevant task(s).
    FIXED: Only activates energy tasks explicitly mentioned by user.
    """
    msg_lower = user_message.lower()

    # Pre-filter: for energy tasks, only include if keywords present
    energy_tasks_to_check = []
    for task, keywords in ENERGY_TASK_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            energy_tasks_to_check.append(task)

    # Always include hospital and traffic tasks in LLM check
    non_energy_tasks = {k: v for k, v in TASK_DESCRIPTIONS.items()
                        if k not in ENERGY_TASK_KEYWORDS}
    tasks_to_check   = dict(non_energy_tasks)
    for t in energy_tasks_to_check:
        tasks_to_check[t] = TASK_DESCRIPTIONS[t]

    # If no energy keywords found at all, still let LLM decide
    if not energy_tasks_to_check:
        tasks_to_check = TASK_DESCRIPTIONS

    task_desc = "\n".join(f"- {k}: {v}" for k, v in tasks_to_check.items())

    history_text = ""
    if conversation_history:
        history_text = "\n\nRecent conversation:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation_history[-4:]
        )

    system_prompt = f"""You are a router for UMORDA — a smart decision system.
Read the user message and decide which task(s) it is DIRECTLY about.

Available tasks:
{task_desc}

STRICT RULES:
- Only include tasks the user EXPLICITLY mentions or strongly implies.
- Do NOT include tasks just because they are related — user must mention them.
- Example: "battery is charged" → only battery_management. NOT solar or grid.
- Example: "solar panels working well" → only solar_scheduling. NOT battery or grid.
- Example: "grid is expensive" → battery_management AND grid_interaction (price affects both).
- If message mentions rain/no sun → solar_scheduling (solar=0 situation).
- Output ONLY: {{"tasks": ["task_name", ...]}}
"""
    user_prompt = f'Message: "{user_message}"{history_text}\n\nOutput JSON only.'

    raw    = _call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=80)
    parsed = _parse_json(raw)
    tasks  = parsed.get("tasks", [])
    return [t for t in tasks if t in TASK_FIELD_SPECS]


# =============================================================================
# 2. EXTRACTOR — FIXED: understands scale and weather context properly
# =============================================================================
def extract_state(task, user_message, known_state, conversation_history=None):
    """
    Extracts numeric state from natural language.
    FIXED: Proper scale understanding, weather context, price mapping.
    """
    fields     = TASK_FIELD_SPECS[task]
    field_desc = "\n".join(f"- {k}: {v}" for k, v in fields.items())

    history_text = ""
    if conversation_history:
        history_text = "\n\nConversation history:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation_history[-4:]
        )

    system_prompt = f"""You are a precise data extractor for UMORDA energy management.
Task: {task} — {TASK_DESCRIPTIONS.get(task, '')}

{SCALE_CONTEXT}

Fields to extract:
{field_desc}

CRITICAL EXTRACTION RULES:
1. Rain / cloudy / overcast / no sun / night → solar_output = 0, solar_surplus = 0
2. "High price" / "expensive grid" / "costly grid" → grid_price = 2
3. "Cheap grid" / "low price" → grid_price = 0
4. "Some charge" / "some battery" → battery_level = 5 (medium)
5. "Low battery" → battery_level = 2
6. "Full battery" → battery_level = 9
7. "A lot of solar" / "panels working well" → solar_output = 7
8. "Weak solar" / "little sun" → solar_output = 2
9. Only update fields the message mentions. Keep others from known state.
10. If value is impossible → needs_clarification = true.

Output ONLY this JSON:
{{
  "state": {{...all fields...}},
  "needs_clarification": false,
  "clarification_question": null,
  "notes": "brief note on what was extracted"
}}
"""
    user_prompt = f"""Known state: {json.dumps(known_state)}{history_text}

User said: "{user_message}"

Extract and output JSON only."""

    raw    = _call_llm(system_prompt, user_prompt, temperature=0.1, max_tokens=300)
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
# 3. EXPLAINER — FIXED: human friendly, no raw Q-values, correct scale words
# =============================================================================
def explain_decision(task, state, action, reason_hint, policy_chunks):
    """
    Explains agent decision in plain human-friendly language.
    FIXED: No raw Q-values, correct scale descriptions, clear reasoning.
    """
    policy_text = "\n\n".join(c["text"] for c in policy_chunks)

    # Convert numeric state to human-friendly descriptions
    def describe_state(task, state):
        desc = []
        if task == "solar_scheduling":
            solar = state.get("solar_output", 0)
            desc.append(f"solar output is {'none' if solar==0 else 'weak' if solar<=3 else 'moderate' if solar<=6 else 'strong'} ({solar}/9)")
            batt  = state.get("battery_level", 0)
            desc.append(f"battery is {'empty' if batt<=1 else 'low' if batt<=3 else 'medium' if batt<=6 else 'high' if batt<=8 else 'full'} ({batt}/9)")
            times = ["morning", "afternoon", "evening", "night"]
            desc.append(f"time is {times[state.get('time_of_day',0)]}")
        elif task == "battery_management":
            batt  = state.get("battery_level", 0)
            desc.append(f"battery is {'empty' if batt<=1 else 'low' if batt<=3 else 'medium' if batt<=6 else 'high' if batt<=8 else 'full'} ({batt}/9)")
            price = state.get("grid_price", 1)
            desc.append(f"grid price is {'cheap' if price==0 else 'normal' if price==1 else 'expensive'}")
            solar = state.get("solar_output", 0)
            desc.append(f"solar is {'unavailable' if solar==0 else 'weak' if solar<=3 else 'available'} ({solar}/9)")
        elif task == "grid_interaction":
            price = state.get("grid_price", 1)
            desc.append(f"grid price is {'cheap' if price==0 else 'normal' if price==1 else 'expensive'}")
            surp  = state.get("solar_surplus", 0)
            desc.append(f"solar surplus is {'none' if surp==0 else 'small' if surp<=3 else 'good' if surp<=6 else 'large'} ({surp}/9)")
            batt  = state.get("battery_level", 0)
            desc.append(f"battery is {'empty' if batt<=1 else 'low' if batt<=3 else 'medium' if batt<=6 else 'high'} ({batt}/9)")
        return ", ".join(desc)

    friendly_state = describe_state(task, state)

    system_prompt = f"""You are a friendly energy advisor explaining a smart home decision.
Write a SHORT clear explanation (2-3 sentences MAX) for a regular home owner.

{SCALE_CONTEXT}

STRICT RULES:
- NEVER mention Q-table values or numbers like "79.593" — these mean nothing to users.
- NEVER say "relatively low" for grid_price=2 — price 2 is EXPENSIVE, always say so!
- NEVER say battery level 5 is "low" — 5/9 is MEDIUM.
- Use simple everyday language — imagine explaining to your neighbor.
- Focus on WHY the decision makes sense in real life.
- Keep it under 3 sentences.
"""

    user_prompt = f"""Situation: {friendly_state}
Decision made: {action}
Internal reason: {reason_hint}

Policy context:
{policy_text}

Explain this decision simply to a home owner. No Q-values. No technical jargon."""

    return _call_llm(system_prompt, user_prompt, temperature=0.3, max_tokens=200)


# =============================================================================
# 4. EXPLAINER WITHOUT POLICY (for comparison)
# =============================================================================
def explain_ungrounded(task, state, action, reason_hint):
    system_prompt = """You are a friendly energy advisor. Explain a smart home 
energy decision in 2-3 simple sentences. No Q-values. No technical terms."""

    user_prompt = f"""Task: {task}
Situation: {json.dumps(state)}
Decision: {action}

Explain simply to a home owner."""

    return _call_llm(system_prompt, user_prompt, temperature=0.3, max_tokens=200)


# =============================================================================
# SELF-TEST — run this file directly to see the ACTIVE backend in action
# using the real router / extractor / explainer functions above.
#   python llm_client.py                 -> uses whatever LLM_BACKEND is set to
#   $env:LLM_BACKEND="ollama"; python llm_client.py   -> forces the 7B model
# =============================================================================
if __name__ == "__main__":
    print("\n" + "#" * 60)
    print(f"#   UMORDA LLM CLIENT SELF-TEST")
    print(f"#   Backend: {LLM_BACKEND}"
          + (f" (model={OLLAMA_MODEL})" if LLM_BACKEND == "ollama" else f" (model={MODEL})"))
    print("#" * 60)

    try:
        # ── Router ────────────────────────────────────────────────────────
        print("\n[1] route_message()")
        msg = "Solar panels are producing a lot of power right now"
        tasks = route_message(msg)
        print(f"    Message: {msg}")
        print(f"    Routed tasks: {tasks}")

        # ── Extractor ─────────────────────────────────────────────────────
        print("\n[2] extract_state()")
        task = tasks[0] if tasks else "solar_scheduling"
        known_state = {k: 0 for k in TASK_FIELD_SPECS[task]}
        result = extract_state(task, msg, known_state)
        print(f"    Task: {task}")
        print(f"    Extracted state: {result['state']}")
        print(f"    Needs clarification: {result['needs_clarification']}")
        print(f"    Notes: {result['notes']}")

        # ── Explainer ─────────────────────────────────────────────────────
        print("\n[3] explain_decision()")
        state       = {"solar_output": 8, "home_consumption": 3,
                       "battery_level": 4, "time_of_day": 1}
        action      = "Use Solar Directly"
        reason_hint = "solar output high, meets and exceeds consumption"
        policy_chunks = [{
            "text": ("When solar output is high and exceeds home consumption, "
                     "use it directly to power the home rather than storing "
                     "it or buying from the grid, since this avoids battery "
                     "conversion losses and reduces grid dependency."),
            "source": "solar_scheduling_policy.md",
        }]
        explanation = explain_decision(task, state, action, reason_hint, policy_chunks)
        print(f"    State: {state}")
        print(f"    Action: {action}")
        print(f"    Explanation:\n    {explanation}")

        print("\n" + "=" * 60)
        print("  SELF-TEST COMPLETE")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
        if LLM_BACKEND == "ollama":
            print("  Check Ollama is running and OLLAMA_MODEL matches `ollama list`.")
        else:
            print("  Check GROQ_API_KEY is set.")