"""
UMORDA — Traffic Domain Assistant
File: traffic_assistant.py

Mirrors Sourav's hospital_assistant.py for the Traffic domain.
Connects: llm_client.py (Groq + Llama 3) + TrafficEnv + TrafficAgent Q-tables

Flow per user message:
  1. route_message()    → which traffic task is this about?
  2. extract_state()    → pull numbers from natural language
  3. Q-table lookup     → best_action() + hard safety override
  4. explain_decision() → turn action into plain English

Usage:
  python traffic_assistant.py
"""

import os
import sys
import json

sys.path.append(os.path.dirname(__file__))

from environments.traffic_env import TrafficEnv
from agents.traffic_agent import TrafficAgent
import llm_client

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "qtables")

# Field order must match TrafficEnv._encode_state() exactly
TASK_STATE_ORDER = {
    "intersection": ["cars_NS", "cars_EW", "current_phase", "phase_elapsed", "wait_NS", "wait_EW"],
    "pedestrian":   ["peds", "vehs", "ped_wait", "veh_wait", "phase", "elapsed"],
    "parking":      ["spots", "incoming", "queue_wait", "occupancy"],
}

# Default starting state for each task when conversation begins
DEFAULT_STATE = {
    "intersection": {"cars_NS": 2, "cars_EW": 2, "current_phase": 0,
                     "phase_elapsed": 0, "wait_NS": 0, "wait_EW": 0},
    "pedestrian":   {"peds": 0, "vehs": 0, "ped_wait": 0,
                     "veh_wait": 0, "phase": 1, "elapsed": 0},
    "parking":      {"spots": 10, "incoming": 0, "queue_wait": 0, "occupancy": 0},
}

# Policy text used by explainer (RAG-grounded explanations)
TRAFFIC_POLICY = {
    "intersection": [
        {"text": "Traffic signals should prioritize the direction with the longer "
                 "cumulative wait time — not just the higher car count — to ensure "
                 "fairness. A hard safety limit forces a switch if any direction "
                 "waits beyond the maximum threshold, regardless of Q-values."},
        {"text": "Minimum green-light durations must be respected to avoid rapid "
                 "signal flipping, which reduces throughput and causes driver confusion."},
    ],
    "pedestrian": [
        {"text": "Pedestrian safety takes absolute priority. If a pedestrian has "
                 "waited beyond the safety threshold, the signal must switch to allow "
                 "crossing — this is a hard guarantee, not a soft preference."},
        {"text": "Below the safety threshold, the system balances pedestrian and "
                 "vehicle wait times to minimize total cumulative delay for both groups."},
    ],
    "parking": [
        {"text": "Parking entry should remain open whenever capacity exists. Entry "
                 "should only close when the lot is near full or the queue has waited "
                 "beyond an acceptable limit with no relief possible."},
    ],
}


# =============================================================================
# TRAFFIC ASSISTANT CLASS
# =============================================================================
class TrafficAssistant:
    """
    Loads all 3 trained Traffic Q-tables and handles natural language input.
    Call handle_message() with any user text to get a decision + explanation.
    """

    def __init__(self):
        self.envs   = {}
        self.agents = {}

        print("\n  Loading Traffic Q-Tables...")
        for task in ["intersection", "pedestrian", "parking"]:
            env   = TrafficEnv(task=task)
            agent = TrafficAgent(
                env.observation_space.n,
                env.action_space.n,
                epsilon=0.0  # no exploration during demo
            )
            path = os.path.join(QTABLE_DIR, f"traffic_{task}_qtable.npy")
            if os.path.exists(path):
                agent.load_qtable(path)
            else:
                print(f"  [!] Q-Table missing for '{task}'.")
                print(f"      Run: python training/train_traffic.py  first!")
            self.envs[task]   = env
            self.agents[task] = agent

        # Live state memory — updated each turn as user describes situation
        self.state   = {t: dict(DEFAULT_STATE[t]) for t in DEFAULT_STATE}
        # Conversation history — for follow-up messages
        self.history = []

    # ------------------------------------------------------------------
    def _decide(self, task, state_dict):
        """Run Q-table + safety override. Returns action label + details."""
        env   = self.envs[task]
        agent = self.agents[task]
        order = TASK_STATE_ORDER[task]
        raw   = [state_dict[k] for k in order]

        encoded        = env._encode_state(raw)
        env._raw_state = raw
        q_action       = agent.best_action(encoded)
        final_action   = env._safety_override(q_action)

        cfg          = TrafficEnv.TASK_CONFIG[task]
        action_label = cfg["action_meanings"][final_action]
        q_values     = agent.q_table[encoded].tolist()
        overridden   = (final_action != q_action)

        if overridden:
            reason = (f"Safety override triggered — raw Q-table preferred "
                      f"'{cfg['action_meanings'][q_action]}' but safety limit was breached")
        else:
            q_named = {cfg["action_meanings"][i]: round(q_values[i], 3)
                       for i in range(len(q_values))}
            reason = f"Q-table values: {q_named}"

        return {
            "action":       action_label,
            "action_index": final_action,
            "q_values":     q_values,
            "overridden":   overridden,
            "reason":       reason,
        }

    # ------------------------------------------------------------------
    def handle_message(self, user_message):
        """
        Full pipeline for one user turn.
        Returns dict with reply text + technical details.
        """
        self.history.append({"role": "user", "content": user_message})

        # STEP 1 — Route: which traffic task(s)?
        all_tasks     = llm_client.route_message(user_message, self.history)
        traffic_tasks = [t for t in all_tasks if t in TASK_STATE_ORDER]

        if not traffic_tasks:
            reply = ("I couldn't identify which traffic situation you're describing. "
                     "Please mention the intersection, pedestrian crossing, or "
                     "parking lot specifically.")
            self.history.append({"role": "assistant", "content": reply})
            return {"tasks": [], "reply": reply, "results": []}

        results = []
        for task in traffic_tasks:

            # STEP 2 — Extract: pull numbers from message
            extraction = llm_client.extract_state(
                task, user_message, self.state[task], self.history
            )

            # If LLM needs clarification, ask user instead of guessing
            if extraction["needs_clarification"]:
                results.append({
                    "task":                 task,
                    "clarification_needed": True,
                    "question":             extraction["clarification_question"],
                })
                continue

            # Update live state memory
            self.state[task] = extraction["state"]

            # STEP 3 — Decide: Q-table + safety override
            decision = self._decide(task, self.state[task])

            # STEP 4 — Explain: natural language using policy context
            explanation = llm_client.explain_decision(
                task,
                self.state[task],
                decision["action"],
                decision["reason"],
                TRAFFIC_POLICY.get(task, []),
            )

            results.append({
                "task":                 task,
                "clarification_needed": False,
                "state":                self.state[task],
                "decision":             decision,
                "explanation":          explanation,
                "notes":                extraction["notes"],
            })

        # Build combined reply
        reply_parts = []
        for r in results:
            if r["clarification_needed"]:
                reply_parts.append(r["question"])
            else:
                reply_parts.append(f"[{r['task'].upper()}] {r['explanation']}")
        reply = "\n\n".join(reply_parts)

        self.history.append({"role": "assistant", "content": reply})
        return {"tasks": traffic_tasks, "results": results, "reply": reply}


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
def main():
    print("\n" + "=" * 65)
    print("  UMORDA — Traffic Assistant")
    print("  Powered by: Q-Learning + Groq API + Llama 3")
    print("=" * 65)
    print("\n  You can type naturally, for example:")
    print('  "There are a lot of cars coming from the north and south"')
    print('  "Pedestrians have been waiting a long time at the crossing"')
    print('  "The parking lot is almost full and cars keep arriving"')
    print('  "North side has 6 cars, east side only 1"')
    print("\n  Type 'state' to see current known state")
    print("  Type 'reset' to reset all states")
    print("  Type 'q' to quit")
    print("=" * 65)

    try:
        assistant = TrafficAssistant()
    except Exception as e:
        print(f"\n  [!] Error loading assistant: {e}\n")
        return

    while True:
        print()
        msg = input("  You: ").strip()

        if not msg:
            continue

        if msg.lower() == "q":
            print("\n  Goodbye! 👋\n")
            break

        if msg.lower() == "state":
            print("\n  Current known states:")
            for task, state in assistant.state.items():
                print(f"    [{task}]: {state}")
            continue

        if msg.lower() == "reset":
            assistant.state   = {t: dict(DEFAULT_STATE[t]) for t in DEFAULT_STATE}
            assistant.history = []
            print("\n  All states reset to defaults. ✅")
            continue

        try:
            result = assistant.handle_message(msg)
        except EnvironmentError as e:
            print(f"\n  [!] API Key Error: {e}\n")
            break
        except Exception as e:
            print(f"\n  [!] Error: {e}\n")
            continue

        # Show natural language reply
        print(f"\n  Assistant: {result['reply']}")

        # Show technical details underneath (great for faculty demo!)
        if result.get("results"):
            for r in result["results"]:
                if not r["clarification_needed"]:
                    print(f"\n  ── Technical Detail [{r['task']}] ──────────────")
                    print(f"  State    : {r['state']}")
                    dec = r["decision"]
                    override_tag = "  ⚠ SAFETY OVERRIDE!" if dec["overridden"] else ""
                    print(f"  Decision : {dec['action']}{override_tag}")
                    print(f"  Q-Values : {[round(v, 3) for v in dec['q_values']]}")
                    print(f"  Notes    : {r['notes']}")
                    print(f"  {'─'*48}")


if __name__ == "__main__":
    main()
