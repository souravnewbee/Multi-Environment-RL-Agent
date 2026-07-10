"""
UMORDA — Energy Domain Assistant (FIXED VERSION)
File: energy_assistant.py

Fixes applied:
- State validation: solar=0 never picks "Use Solar Directly"
- Router only activates tasks user actually mentioned
- Explainer uses human friendly language
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
        {"text": "Solar power should be used directly when it meets or exceeds home "
                 "consumption. Surplus solar should be stored in the battery for evening "
                 "and night use. When solar output is zero (rain, night), the battery "
                 "should be used first before buying from the grid."},
    ],
    "battery_management": [
        {"text": "Charge the battery when solar is strong and battery has space. "
                 "Discharge the battery when grid price is expensive (price=2) to avoid "
                 "costly purchases. Keep battery idle when solar already covers home needs."},
    ],
    "grid_interaction": [
        {"text": "Sell to the grid when solar surplus is high and grid price is expensive "
                 "— this earns maximum money. Stay self-sufficient when own solar and "
                 "battery covers consumption. Only buy from grid when own resources are "
                 "fully depleted, and prefer cheap tariff periods."},
    ],
}


class EnergyAssistant:

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
                print(f"  [!] Q-Table missing for '{task}'. Run train_energy.py first!")
            self.envs[task]   = env
            self.agents[task] = agent

        self.state   = {t: dict(DEFAULT_STATE[t]) for t in DEFAULT_STATE}
        self.history = []

    # ------------------------------------------------------------------
    # FIX: State validation before Q-table lookup
    # ------------------------------------------------------------------
    def _validate_state(self, task, state_dict):
        """
        Fix impossible state situations before querying Q-table.
        Example: solar=0 should never pick 'Use Solar Directly'
        """
        issues = []

        if task == "solar_scheduling":
            if state_dict.get("solar_output", 0) == 0:
                issues.append("No solar available (solar_output=0)")

        elif task == "grid_interaction":
            if state_dict.get("solar_surplus", 0) == 0:
                issues.append("No solar surplus available (solar_surplus=0)")

        return issues

    # ------------------------------------------------------------------
    # Q-table decision with validation
    # ------------------------------------------------------------------
    def _decide(self, task, state_dict):
        env   = self.envs[task]
        agent = self.agents[task]
        order = TASK_STATE_ORDER[task]
        raw   = [state_dict[k] for k in order]

        encoded        = env._encode_state(raw)
        env._raw_state = raw
        cfg            = EnergyEnv.TASK_CONFIG[task]

        # Get Q-table best action
        q_action     = agent.best_action(encoded)
        action_label = cfg["action_meanings"][q_action]
        q_values     = agent.q_table[encoded].tolist()

        # FIX: Override impossible actions
        override_reason = None

        if task == "solar_scheduling":
            solar = state_dict.get("solar_output", 0)
            if solar == 0 and q_action == 0:  # "Use Solar Directly" when solar=0
                # Override to battery or grid
                batt = state_dict.get("battery_level", 0)
                q_action     = 2 if batt <= 1 else 1  # buy grid if empty, else use battery
                action_label = cfg["action_meanings"][q_action]
                override_reason = f"Solar=0 so cannot use solar directly. Using {'battery' if q_action==1 else 'grid'} instead."

        elif task == "grid_interaction":
            surplus = state_dict.get("solar_surplus", 0)
            if surplus == 0 and q_action == 1:  # "Sell to Grid" when no surplus
                batt = state_dict.get("battery_level", 0)
                cons = state_dict.get("home_consumption", 3)
                q_action     = 2 if batt >= cons else 0
                action_label = cfg["action_meanings"][q_action]
                override_reason = f"No surplus to sell. Using {'self-sufficient' if q_action==2 else 'grid'} instead."

        # Build reason string
        price_words = ["cheap", "normal", "expensive"]
        if task == "battery_management":
            price = state_dict.get("grid_price", 1)
            batt  = state_dict.get("battery_level", 5)
            batt_word  = "empty" if batt<=1 else "low" if batt<=3 else "medium" if batt<=6 else "high"
            price_word = price_words[price]
            reason = f"Battery is {batt_word}, grid is {price_word}"
        elif task == "solar_scheduling":
            solar = state_dict.get("solar_output", 0)
            solar_word = "unavailable" if solar==0 else "weak" if solar<=3 else "moderate" if solar<=6 else "strong"
            reason = f"Solar is {solar_word} ({solar}/9)"
        elif task == "grid_interaction":
            surplus = state_dict.get("solar_surplus", 0)
            price   = state_dict.get("grid_price", 1)
            reason  = f"Surplus={surplus}/9, grid is {price_words[price]}"
        else:
            reason = "Q-table based decision"

        if override_reason:
            reason = f"[Override] {override_reason}"

        return {
            "action":          action_label,
            "action_index":    q_action,
            "q_values":        q_values,
            "reason":          reason,
            "was_overridden":  override_reason is not None,
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def handle_message(self, user_message):
        self.history.append({"role": "user", "content": user_message})

        # STEP 1 — Route (only tasks user mentioned)
        all_tasks    = llm_client.route_message(user_message, self.history)
        energy_tasks = [t for t in all_tasks if t in TASK_STATE_ORDER]

        if not energy_tasks:
            reply = ("I couldn't identify which energy situation you're describing. "
                     "Please mention solar panels, battery, or grid electricity.")
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

            # STEP 3 — Validate + Decide
            issues   = self._validate_state(task, self.state[task])
            decision = self._decide(task, self.state[task])

            # STEP 4 — Explain (human friendly)
            explanation = llm_client.explain_decision(
                task, self.state[task], decision["action"],
                decision["reason"], ENERGY_POLICY.get(task, []),
            )

            results.append({
                "task":                 task,
                "clarification_needed": False,
                "state":                self.state[task],
                "decision":             decision,
                "explanation":          explanation,
                "notes":                extraction["notes"],
                "validation_issues":    issues,
            })

        # Build reply
        reply_parts = []
        for r in results:
            if r["clarification_needed"]:
                reply_parts.append(r["question"])
            else:
                override_tag = " ⚠ (Corrected)" if r["decision"]["was_overridden"] else ""
                reply_parts.append(f"[{r['task'].upper()}]{override_tag}\n{r['explanation']}")
        reply = "\n\n".join(reply_parts)

        self.history.append({"role": "assistant", "content": reply})
        return {"tasks": energy_tasks, "results": results, "reply": reply}


# =============================================================================
# CLI
# =============================================================================
def main():
    print("\n" + "="*65)
    print("  UMORDA — Energy Assistant (FIXED VERSION)")
    print("  Balcony Solar Panel Optimization")
    print("  Powered by: Q-Learning + Groq API + Llama 3")
    print("="*65)
    print("\n  Try typing:")
    print('  "My solar is not generating much, grid price is high"')
    print('  "Today is raining, no sun, grid is expensive"')
    print('  "Battery has some charge, grid price is high"')
    print('  "Afternoon, lots of sunshine, battery half full"')
    print("\n  Commands: 'state', 'reset', 'q'")
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

        print(f"\n  Assistant:\n  {result['reply']}")

        if result.get("results"):
            for r in result["results"]:
                if not r["clarification_needed"]:
                    print(f"\n  ── [{r['task']}] Technical Detail ──")
                    # Human friendly state description
                    state = r["state"]
                    if r["task"] == "solar_scheduling":
                        solar = state["solar_output"]
                        print(f"  Solar  : {'None' if solar==0 else 'Weak' if solar<=3 else 'Moderate' if solar<=6 else 'Strong'} ({solar}/9)")
                        batt  = state["battery_level"]
                        print(f"  Battery: {'Empty' if batt<=1 else 'Low' if batt<=3 else 'Medium' if batt<=6 else 'High'} ({batt}/9)")
                    elif r["task"] == "battery_management":
                        batt  = state["battery_level"]
                        price = state["grid_price"]
                        print(f"  Battery: {'Empty' if batt<=1 else 'Low' if batt<=3 else 'Medium' if batt<=6 else 'High'} ({batt}/9)")
                        print(f"  Grid   : {'Cheap' if price==0 else 'Normal' if price==1 else 'Expensive'}")
                    elif r["task"] == "grid_interaction":
                        price  = state["grid_price"]
                        surp   = state["solar_surplus"]
                        print(f"  Grid   : {'Cheap' if price==0 else 'Normal' if price==1 else 'Expensive'}")
                        print(f"  Surplus: {'None' if surp==0 else 'Small' if surp<=3 else 'Good' if surp<=6 else 'Large'} ({surp}/9)")
                    dec = r["decision"]
                    override = " ⚠ OVERRIDDEN" if dec["was_overridden"] else ""
                    print(f"  Decision: {dec['action']}{override}")
                    print(f"  {'─'*45}")


if __name__ == "__main__":
    main()
