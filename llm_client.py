"""
UMORDA — Groq LLM Integration
Responsibilities, each its own focused prompt:

  1. route_message()     — given a free-form user message, decide which
                            hospital task(s) it concerns (multi-task aware).

  2. extract_state()     — turn a user's natural language message into the
                            structured numeric fields the Q-table needs,
                            using conversation history + known live values
                            as context. Flags when input is implausible or
                            too vague to extract safely, instead of guessing.

  3. explain_decision()  — turn the RL agent's chosen action into a
                            natural-language explanation, grounded in a
                            retrieved policy passage (RAG).

  4. explain_ungrounded() — same as above but WITHOUT policy context, for
                            side-by-side comparison showing why RAG matters.

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
        raise ValueError(f"LLM did not return valid JSON. Raw output: {raw}")


# ── Task field specs (shared across router + extractor) ──────────────────────
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

TASK_DESCRIPTIONS = {
    "bed_allocation":   "managing hospital beds — admitting, transferring, or rejecting patients based on bed capacity",
    "er_queue":         "emergency room triage — deciding whether to serve emergency patients or normal-queue patients",
    "staff_allocation": "doctor staffing levels — assigning more doctors, keeping current staff, or reducing staff based on patient load",
}


# ── 1. Router: which task(s) does this message concern? ──────────────────────
def route_message(user_message, conversation_history=None):
    """
    Decide which hospital task(s) a free-form message concerns.
    Supports multi-task messages (e.g. "ER is backed up and we're low on doctors"
    routes to BOTH er_queue and staff_allocation).

    Parameters
    ----------
    user_message          : the latest free-form message from the user
    conversation_history   : optional list of {"role": "user"/"assistant", "content": ...}
                              for context on follow-up messages

    Returns
    -------
    list of task names from ["bed_allocation", "er_queue", "staff_allocation"]
    (empty list if the message doesn't clearly concern any of them)
    """
    task_desc = "\n".join(f"- {k}: {v}" for k, v in TASK_DESCRIPTIONS.items())
    history_text = ""
    if conversation_history:
        history_text = "\n\nRecent conversation:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation_history[-6:]
        )

    system_prompt = f"""You are a router for a hospital resource management system.
Read the staff member's message and decide which of these tasks it concerns:

{task_desc}

Rules:
- A message can concern MULTIPLE tasks at once (e.g. mentioning both ER queue
  and doctor staffing). Include all that apply.
- If the message is a continuation/follow-up of a previous topic (check the
  conversation history), route it to the same task(s) as that topic unless
  it clearly shifts topic.
- If the message doesn't concern any hospital task (e.g. small talk, unrelated
  question), return an empty list.
- Output ONLY a JSON object: {{"tasks": ["task_name", ...]}}. No other text.
"""

    user_prompt = f"Staff message: \"{user_message}\"{history_text}\n\nOutput JSON."

    raw = _call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=100)
    parsed = _parse_json(raw)
    tasks = parsed.get("tasks", [])
    return [t for t in tasks if t in TASK_FIELD_SPECS]


# ── 2. Extractor with plausibility check + conversation memory ───────────────
def extract_state(task, user_message, known_state, conversation_history=None):
    """
    Parameters
    ----------
    task                  : one of "bed_allocation", "er_queue", "staff_allocation"
    user_message          : natural language description from the user
    known_state           : dict of currently known live values, used as defaults
    conversation_history  : optional list of {"role", "content"} for follow-up context

    Returns
    -------
    dict with keys:
      "state"          : extracted/estimated state dict (same keys as known_state)
      "needs_clarification" : bool
      "clarification_question" : str or None — what to ask the user if unclear/implausible
      "notes"          : short string on what was inferred vs given (for transparency)
    """
    fields = TASK_FIELD_SPECS[task]
    field_desc = "\n".join(f"- {k}: {v}" for k, v in fields.items())

    history_text = ""
    if conversation_history:
        history_text = "\n\nRecent conversation (for context on follow-ups):\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in conversation_history[-6:]
        )

    system_prompt = f"""You are a data extraction assistant for a hospital resource
management system. Read a staff member's message and the hospital's current known
numbers, then output an updated state.

Fields required:
{field_desc}

Rules:
- Start from the known current values given to you.
- If the message implies a change (new arrivals, beds freed up, etc.), update
  the relevant field with your best estimate.
- Vague language should be converted to a reasonable numeric estimate
  (e.g. "a lot of patients" -> a high number within the valid range,
  "a few patients" -> a small number like 2-4).
- If the message is a follow-up (e.g. "a few more arrived"), interpret it as
  ADDING TO the known value, not replacing it.
- If the message gives no information about a field, KEEP the known value unchanged.
- If the message contains a value that is IMPOSSIBLE or far outside realistic
  range for a hospital (e.g. "10000 patients waiting", "-5 beds", "500 doctors"),
  do NOT silently clamp or guess — instead set needs_clarification to true and
  write a short, natural clarifying question to ask the staff member.
- If the message is too vague to extract ANY relevant field (e.g. unrelated
  small talk), also set needs_clarification to true with an appropriate question.
- Output ONLY a JSON object with this exact shape, no other text:
  {{
    "state": {{ ...fields... }},
    "needs_clarification": true/false,
    "clarification_question": "..." or null,
    "notes": "short note on what was inferred vs taken directly from the message"
  }}
"""

    user_prompt = f"""Known current state: {json.dumps(known_state)}{history_text}

Staff message: "{user_message}"

Output the JSON response."""

    raw = _call_llm(system_prompt, user_prompt, temperature=0.2, max_tokens=350)
    parsed = _parse_json(raw)

    result_state = dict(known_state)
    extracted_fields = parsed.get("state", {})
    for key in fields.keys():
        if key in extracted_fields:
            try:
                result_state[key] = int(extracted_fields[key])
            except (TypeError, ValueError):
                pass

    return {
        "state":                   result_state,
        "needs_clarification":     bool(parsed.get("needs_clarification", False)),
        "clarification_question":  parsed.get("clarification_question"),
        "notes":                   parsed.get("notes", ""),
    }


# ── 3. Action → natural language explanation (RAG-grounded) ──────────────────
def explain_decision(task, state, action, reason_hint, policy_chunks):
    """Grounded explanation using retrieved policy text (RAG)."""
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

    return _call_llm(system_prompt, user_prompt, temperature=0.4, max_tokens=250)


# ── 4. Action → explanation WITHOUT RAG (for comparison) ─────────────────────
def explain_ungrounded(task, state, action, reason_hint):
    """
    Same explanation task as explain_decision(), but with NO policy context
    given. Used to demonstrate what the LLM does when it has to invent its
    own justification — useful for showing why RAG grounding matters.
    """
    system_prompt = """You are an assistant that explains hospital resource management
decisions made by a reinforcement learning system. Write a clear, professional
2-4 sentence explanation for hospital staff. You have NOT been given any
official policy documents — explain the decision using your own general
knowledge of hospital operations."""

    user_prompt = f"""Task: {task}
Current situation: {json.dumps(state)}
Decision made: {action}
Internal reason: {reason_hint}

Explain this decision to hospital staff in plain language."""

    return _call_llm(system_prompt, user_prompt, temperature=0.4, max_tokens=250)


if __name__ == "__main__":
    # Smoke tests (require GROQ_API_KEY to be set)
    print("── Router test ──")
    tasks = route_message("ER is backed up and we're also low on doctors today")
    print("Routed tasks:", tasks)

    print("\n── Extractor test (normal) ──")
    known = {"free_beds": 8, "waiting_patients": 0}
    result = extract_state(
        "bed_allocation",
        "We've got a flood of patients coming in, beds are filling up fast",
        known,
    )
    print(result)

    print("\n── Extractor test (implausible input) ──")
    result2 = extract_state(
        "bed_allocation",
        "We have 10000 patients waiting right now",
        known,
    )
    print(result2)