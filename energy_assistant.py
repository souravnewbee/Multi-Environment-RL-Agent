"""
UMORDA — Energy Domain Assistant
File: energy_assistant.py

Mirrors traffic_assistant.py and hospital_assistant.py for the Energy domain.
Connects: llm_client.py (Groq + Llama 3) + EnergyEnv + EnergyAgent Q-tables

Domain: Balcony Solar Panel (Balkonkraftwerk) Optimization

Flow per user message:
  1. route_message()    → which energy task is this about?
  2. extract_state()    → pull numbers from natural language
  3. Q-table lookup     → best_action()
  4. explain_decision() → turn action into plain English

Usage:
  python energy_assistant.py
"""

import os, sys, json
sys.path.append(os.path.dirname(__file__))

from environments.energy_env import EnergyEnv
from agents.energy_agent import EnergyAgent
import llm_client

QTABLE_DIR = os.path.join(os.path.dirname(__file__), "qtables")

TASK_STATE_ORDER = {
    "solar_scheduling":   ["solar_output", "home_consumption", "battery_level", "time_of_day"],
    "battery_management": ["battery_level", "solar_output", "grid_price", "home_consumption"],
    "grid_interaction":   ["grid_price", "solar_surplus", "battery_level", "home_consumption"],
}

DEFAULT_STATE = {
    "solar_scheduling":   {"solar_output": 0, "home_consumption": 3, "battery_level": 5, "time_of_day": 1},
    "battery_management": {"battery_level": 5, "solar_output": 0, "grid_price": 1, "home_consumption": 3},
    "grid_interaction":   {"grid_price": 1, "solar_surplus": 0, "battery_level": 5, "home_consumption": 3},
}

ENERGY_POLICY = {
    "solar_scheduling": [
        {"text": "Solar power should be used directly whenever it matches or exceeds "
                 "home consumption — this avoids unnecessary conversion losses. "
                 "Surplus solar should be stored in the battery for evening and night use "
                 "rather than wasted. Grid electricity should only be purchased when "
                 "solar output is zero and the battery is depleted."},
    ],
    "battery_management": [
        {"text": "The battery should be charged when solar generation is strong and "
                 "the battery has capacity. Discharging is most beneficial when grid "
                 "prices are high (expensive tariff period), so stored energy displaces "
                 "costly grid purchases. Keeping the battery idle is appropriate when "
                 "solar covers consumption without battery involvement."},
    ],
    "grid_interaction": [
        {"text": "When solar surplus is high and grid prices are expensive, selling "
                 "electricity back to the grid maximizes financial return. Self-sufficiency "
                 "should be the default when the home's own solar and battery can cover "
                 "consumption. Buying from the grid is only justified when own resources "
                 "are fully depleted, and is best done during cheap tariff periods."},
    ],
}


class EnergyAssistant:
    """
    Loads all 3 trained Energy Q-tables and handles natural language input
    about balcony solar panel optimization.
    """

    def __init__(self):
        self.envs   = {}
        self.agents = {}

        print("\n  Loading Energy Q-Tables...")
        for task in ["solar_scheduling", "battery_management", "grid_interaction"]:
            env   = EnergyEnv(task=task)
            agent = EnergyAgent(env.observation_space.n, env.action_space.n, epsilon=0.0)
            path  = os.path.join(QTABLE_DIR, f"energy_{task}_qtable.npy")
            if os.path.exists(path):
                agent.load_qtable(path)
            else:
                print(f"  [!] Q-Table missing for '{task}'.")
                print(f"      Run: python training/train_energy.py  first!")
            self.envs[task]   = env
            self.agents[task] = agent

        self.state   = {t: dict(DEFAULT_STATE[t]) for t in DEFAULT_STATE}
        self.history = []

    def _decide(self, task, state_dict):
        env    = self.envs[task]
        agent  = self.agents[task]
        order  = TASK_STATE_ORDER[task]
        raw    = [state_dict[k] for k in order]

        encoded        = env._encode_state(raw)
        env._raw_state = raw
        action         = agent.best_action(encoded)
        cfg            = EnergyEnv.TASK_CONFIG[task]
        action_label   = cfg["action_meanings"][action]
        q_values       = agent.q_table[encoded].tolist()

        q_named = {cfg["action_meanings"][i]: round(q_values[i], 3)
                   for i in range(len(q_values))}
        reason  = f"Q-table values: {q_named}"

        return {
            "action":       action_label,
            "action_index": action,
            "q_values":     q_values,
            "reason":       reason,
        }

    def handle_message(self, user_message):
        self.history.append({"role": "user", "content": user_message})

        # STEP 1 — Route
        all_tasks     = llm_client.route_message(user_message, self.history)
        energy_tasks  = [t for t in all_tasks if t in TASK_STATE_ORDER]

        if not energy_tasks:
            reply = ("I couldn't identify which energy situation you're describing. "
                     "Please mention solar panels, battery storage, or grid electricity.")
            self.history.append({"role": "assistant", "content": reply})
            return {"tasks": [], "reply": reply, "results": []}

        results = []
        for task in energy_tasks:

            # STEP 2 — Extract
            extraction = llm_client.extract_state(
                task, user_message, self.state[task], self.history
            )

            if extraction["needs_clarification"]:
                results.append({
                    "task": task, "clarification_needed": True,
                    "question": extraction["clarification_question"],
                })
                continue

            self.state[task] = extraction["state"]

            # STEP 3 — Decide
            decision = self._decide(task, self.state[task])

            # STEP 4 — Explain
            explanation = llm_client.explain_decision(
                task, self.state[task], decision["action"],
                decision["reason"], ENERGY_POLICY.get(task, []),
            )

            results.append({
                "task": task, "clarification_needed": False,
                "state": self.state[task], "decision": decision,
                "explanation": explanation, "notes": extraction["notes"],
            })

        reply_parts = []
        for r in results:
            if r["clarification_needed"]:
                reply_parts.append(r["question"])
            else:
                reply_parts.append(f"[{r['task'].upper()}] {r['explanation']}")
        reply = "\n\n".join(reply_parts)
        self.history.append({"role": "assistant", "content": reply})
        return {"tasks": energy_tasks, "results": results, "reply": reply}


def main():
    print("\n" + "="*65)
    print("  UMORDA — Energy Assistant")
    print("  Balcony Solar Panel (Balkonkraftwerk) Optimization")
    print("  Powered by: Q-Learning + Groq API + Llama 3")
    print("="*65)
    print("\n  Try typing naturally, for example:")
    print('  "My solar panels are generating a lot right now"')
    print('  "Battery is almost full and grid is expensive"')
    print('  "It is afternoon and I have lots of surplus solar"')
    print('  "Battery is empty and there is no sun tonight"')
    print("\n  Type 'state' to see current known state")
    print("  Type 'reset' to reset all states")
    print("  Type 'q' to quit")
    print("="*65)

    try:
        assistant = EnergyAssistant()
    except Exception as e:
        print(f"\n  [!] Error: {e}\n"); return

    while True:
        print()
        msg = input("  You: ").strip()
        if not msg: continue
        if msg.lower() == "q":
            print("\n  Goodbye! 👋\n"); break
        if msg.lower() == "state":
            print("\n  Current known states:")
            for task, state in assistant.state.items():
                print(f"    [{task}]: {state}")
            continue
        if msg.lower() == "reset":
            assistant.state   = {t: dict(DEFAULT_STATE[t]) for t in DEFAULT_STATE}
            assistant.history = []
            print("\n  All states reset. ✅"); continue

        try:
            result = assistant.handle_message(msg)
        except EnvironmentError as e:
            print(f"\n  [!] API Key Error: {e}\n"); break
        except Exception as e:
            print(f"\n  [!] Error: {e}\n"); continue

        print(f"\n  Assistant: {result['reply']}")

        if result.get("results"):
            for r in result["results"]:
                if not r["clarification_needed"]:
                    print(f"\n  ── Technical Detail [{r['task']}] ──")
                    print(f"  State    : {r['state']}")
                    print(f"  Decision : {r['decision']['action']}")
                    print(f"  Q-Values : {[round(v,3) for v in r['decision']['q_values']]}")
                    print(f"  Notes    : {r['notes']}")
                    print(f"  {'─'*50}")


if __name__ == "__main__":
    main()
