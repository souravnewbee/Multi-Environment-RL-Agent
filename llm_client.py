"""
UMORDA — Groq LLM Integration
Two responsibilities, two separate prompts:

  1. extract_state()  — turn a user's natural language message into the
                         structured numeric fields the Q-table needs,
                         using known live values as defaults/context.

  2. explain_decision() — turn the RL agent's chosen action into a
                           natural-language explanation, grounded in a
                           retrieved policy passage (RAG).

Requires: pip install groq
Requires: environment variable GROQ_API_KEY set (get one free at console.groq.com)
"""

import os
import json
from groq import Groq

MODEL = "llama-3.3-70b-versatile"   # fast + free-tier friendly on Groq


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "then run: setx GROQ_API_KEY \"your-key-here\"  (Windows)\n"
            "or: export GROQ_API_KEY=\"your-key-here\"  (Mac/Linux)"
        )
    return Groq(api_key=api_key)


# ── 1. Natural language → structured state ───────────────────────────────────
TASK_FIELD_SPECS = {
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
        "patient_load":      "current patient load / demand on staff (integer, 0-50)",
    },
}


def extract_state(task, user_message, known_state):
    """
    Parameters
    ----------
    task         : one of "bed_allocation", "er_queue", "staff_allocation"
    user_message : natural language description from the user
    known_state   : dict of currently known live values (from hospital_state.json),
                     used as defaults the LLM can override if the message implies a change

    Returns
    -------
    dict with the same keys as known_state, filled in / estimated from the message.
    Also returns "_reasoning" key with a short note on what was inferred vs given.
    """
    client = _get_client()
    fields = TASK_FIELD_SPECS[task]

    field_desc = "\n".join(f"- {k}: {v}" for k, v in fields.items())

    system_prompt = f"""You are a data extraction assistant for a hospital resource
management system. Your ONLY job is to read a staff member's message and the
hospital's current known numbers, then output an updated JSON state.

Fields required:
{field_desc}

Rules:
- Start from the known current values given to you.
- If the message implies a change (new arrivals, beds freed up, etc.), update
  the relevant field with your best estimate.
- Vague language should be converted to a reasonable numeric estimate
  (e.g. "a lot of patients" → a high number within the valid range,
  "a few patients" → a small number like 2-4).
- If the message gives no information about a field, KEEP the known value unchanged.
- Output ONLY valid JSON, no explanation, no markdown formatting, no preamble.
- JSON must have exactly these keys: {list(fields.keys())}
"""

    user_prompt = f"""Known current state: {json.dumps(known_state)}

Staff message: "{user_message}"

Output the updated state as JSON."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"LLM did not return valid JSON. Raw output: {raw}")

    # Safety: ensure all expected keys exist, fall back to known values
    result = dict(known_state)
    for key in fields.keys():
        if key in extracted:
            result[key] = int(extracted[key])

    return result


# ── 2. Action → natural language explanation (RAG-grounded) ──────────────────
def explain_decision(task, state, action, reason_hint, policy_chunks):
    """
    Parameters
    ----------
    task           : task name
    state          : the state the decision was based on
    action         : the action chosen by the Q-table
    reason_hint    : short internal reason string (from hospital_env logic)
    policy_chunks  : list of retrieved policy text chunks (from PolicyRetriever)

    Returns
    -------
    str — natural language explanation for hospital staff
    """
    client = _get_client()

    policy_text = "\n\n".join(c["text"] for c in policy_chunks)

    system_prompt = """You are an assistant that explains hospital resource management
decisions made by a reinforcement learning system. Write a clear, professional
2-4 sentence explanation for hospital staff. Ground your explanation in the
provided policy text — refer to it naturally, don't just quote it. Do not
invent numbers or policies not given to you. Be direct and practical."""

    user_prompt = f"""Task: {task}
Current situation: {json.dumps(state)}
Decision made: {action}
Internal reason: {reason_hint}

Relevant hospital policy:
{policy_text}

Explain this decision to hospital staff in plain language."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=250,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Smoke test (requires GROQ_API_KEY to be set)
    known = {"free_beds": 8, "waiting_patients": 0}
    result = extract_state(
        "bed_allocation",
        "We've got a flood of patients coming in, beds are filling up fast",
        known,
    )
    print("Extracted state:", result)