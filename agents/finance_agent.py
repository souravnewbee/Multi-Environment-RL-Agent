"""
agents/finance_agent.py
========================
Q-Learning agent for the Finance domain.
Matches the exact structure of agents/hospital_agent.py.

Responsibilities:
  - Load trained Q-tables from qtables/
  - Discretise raw state dicts into Q-table indices
  - Select optimal actions via argmax (greedy policy)
  - Return Q-values for all actions (used by explainer.py)
  - Expose metadata about each task

Usage:
    from agents.finance_agent import FinanceAgent

    agent = FinanceAgent(task="trading")
    action, q_values, action_name = agent.get_action(state)
"""

import os
import json
import numpy as np


# ── Task configuration ────────────────────────────────────────────────────────
TASK_CONFIG = {
    "trading": {
        "actions":    ["Buy", "Sell", "Hold"],
        "state_vars": ["price_trend", "shares_held", "cash", "portfolio_value"],
        "q_shape":    (5, 4, 6, 3),
        "description": "Buy/Sell/Hold trading agent using real market price data",
        "objectives":  ["cost", "performance", "fairness"],
    },
    "savings": {
        "actions":    ["Save More", "Spend Normal", "Invest"],
        "state_vars": ["monthly_income", "current_savings", "expenses", "months_remaining"],
        "q_shape":    (6, 5, 5, 3),
        "description": "Personal savings management over a 12-month planning horizon",
        "objectives":  ["cost", "performance", "fairness"],
    },
    "budget": {
        "actions":    ["Allocate Full", "Allocate Partial", "Defer"],
        "state_vars": ["total_budget", "amount_spent", "urgent_requests", "departments_remaining"],
        "q_shape":    (5, 5, 5, 3),
        "description": "Organisational budget allocation across departments with urgent prioritisation",
        "objectives":  ["cost", "performance", "fairness"],
    },
}


class FinanceAgent:
    """
    Trained Q-learning agent for the Finance domain.

    Loads a pre-trained Q-table from disk and provides:
      - get_action(state)       → optimal action index, Q-values, action name
      - get_recommendation(...) → full recommendation dict with weights applied
      - get_all_q_values(state) → raw Q-values for all actions
    """

    QTABLE_DIR = "qtables"

    def __init__(self, task: str = "trading"):
        if task not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task '{task}'. "
                f"Choose from: {list(TASK_CONFIG.keys())}"
            )

        self.task    = task
        self.config  = TASK_CONFIG[task]
        self.actions = self.config["actions"]
        self.n_actions = len(self.actions)

        # Load Q-table
        self.Q        = self._load_qtable()
        self.metadata = self._load_metadata()

    # ── Q-table loading ───────────────────────────────────────────────────────
    def _load_qtable(self) -> np.ndarray:
        path = os.path.join(self.QTABLE_DIR, f"finance_{self.task}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Q-table not found at '{path}'.\n"
                f"Run:  python training/train_finance.py  first."
            )
        Q = np.load(path)
        assert Q.shape == self.config["q_shape"], (
            f"Q-table shape mismatch. "
            f"Expected {self.config['q_shape']}, got {Q.shape}."
        )
        return Q

    def _load_metadata(self) -> dict:
        path = os.path.join(self.QTABLE_DIR, f"finance_{self.task}_metadata.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    # ── State discretisation (must match train_finance.py exactly) ────────────
    def _discretize(self, state: dict) -> tuple:
        if self.task == "trading":
            trend  = int(state.get("price_trend", 0)) + 2
            shares = min(int(state.get("shares_held", 0)), 15) // 4
            cash   = state.get("cash", 0)
            cash_buckets = [0, 200, 500, 1000, 1500, 2000]
            cash_idx = next(
                (i for i, b in enumerate(cash_buckets) if cash <= b), 5
            )
            return (
                int(np.clip(trend,    0, 4)),
                int(np.clip(shares,   0, 3)),
                int(np.clip(cash_idx, 0, 5)),
            )

        elif self.task == "savings":
            income  = min(int(state.get("monthly_income",   0)), 150) // 30
            savings = min(int(state.get("current_savings",  0)), 500) // 125
            months  = min(int(state.get("months_remaining", 0)), 12)  // 3
            return (
                int(np.clip(income,  0, 5)),
                int(np.clip(savings, 0, 4)),
                int(np.clip(months,  0, 4)),
            )

        elif self.task == "budget":
            total    = int(state.get("total_budget",          1))
            spent    = int(state.get("amount_spent",          0))
            used_pct = min(int((spent / max(total, 1)) * 5),  4)
            urgent   = min(int(state.get("urgent_requests",   0)), 4)
            depts    = min(int(state.get("departments_remaining", 0)), 4)
            return (
                int(np.clip(used_pct, 0, 4)),
                int(np.clip(urgent,   0, 4)),
                int(np.clip(depts,    0, 4)),
            )

    # ── Core decision method ──────────────────────────────────────────────────
    def get_action(self, state: dict) -> tuple[int, np.ndarray, str]:
        """
        Given a raw state dict, return:
          action_index  (int)        -- index of the best action
          q_values      (np.ndarray) -- Q-values for all actions
          action_name   (str)        -- human-readable action label

        Example:
            action_idx, q_vals, name = agent.get_action(state)
        """
        s        = self._discretize(state)
        q_values = self.Q[s].copy()
        action   = int(np.argmax(q_values))
        return action, q_values, self.actions[action]

    # ── Multi-objective weighted recommendation ───────────────────────────────
    def get_recommendation(
        self,
        state: dict,
        weights: dict = None,
        step_info: dict = None,
    ) -> dict:
        """
        Returns a full recommendation dict including:
          - best action and its name
          - Q-values for all actions
          - weighted objective scores (if step_info provided)
          - confidence score (gap between best and second-best Q-value)
          - task metadata

        Parameters
        ----------
        state       : current environment state dict
        weights     : {"cost": float, "performance": float, "fairness": float}
                      Must sum to 1.0. Defaults to equal weights.
        step_info   : info dict returned by env.step() — contains
                      r_performance, r_cost, r_fairness if available

        Returns
        -------
        dict with keys: action, action_name, q_values, confidence,
                        weighted_score, objectives, task, state
        """
        if weights is None:
            weights = {"cost": 0.33, "performance": 0.34, "fairness": 0.33}

        action_idx, q_values, action_name = self.get_action(state)

        # Confidence: gap between best and second-best Q-value
        sorted_q   = np.sort(q_values)[::-1]
        confidence = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else 0.0

        # Objective scores from last step (if available)
        objectives = {}
        weighted_score = None
        if step_info:
            r_perf  = step_info.get("r_performance", 0.0)
            r_cost  = step_info.get("r_cost",        0.0)
            r_fair  = step_info.get("r_fairness",    0.0)
            objectives = {
                "performance": round(float(r_perf), 4),
                "cost":        round(float(r_cost),  4),
                "fairness":    round(float(r_fair),  4),
            }
            weighted_score = round(
                weights["cost"]        * r_cost +
                weights["performance"] * r_perf +
                weights["fairness"]    * r_fair,
                4
            )

        # Build all-actions comparison table
        all_actions = [
            {
                "index":  i,
                "name":   self.actions[i],
                "q_value": round(float(q_values[i]), 4),
                "chosen": i == action_idx,
            }
            for i in range(self.n_actions)
        ]

        return {
            "task":           self.task,
            "action":         action_idx,
            "action_name":    action_name,
            "q_values":       [round(float(q), 4) for q in q_values],
            "all_actions":    all_actions,
            "confidence":     round(confidence, 4),
            "weighted_score": weighted_score,
            "objectives":     objectives,
            "weights_used":   weights,
            "state":          state,
            "description":    self.config["description"],
        }

    # ── Convenience methods ───────────────────────────────────────────────────
    def get_all_q_values(self, state: dict) -> dict:
        """Returns a dict mapping action name → Q-value."""
        _, q_values, _ = self.get_action(state)
        return {
            name: round(float(q), 4)
            for name, q in zip(self.actions, q_values)
        }

    def is_confident(self, state: dict, threshold: float = 2.0) -> bool:
        """
        Returns True if the agent is confident in its recommendation.
        Confidence = gap between best and second-best Q-value.
        """
        _, q_values, _ = self.get_action(state)
        sorted_q = np.sort(q_values)[::-1]
        gap = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else 0.0
        return gap >= threshold

    def get_info(self) -> dict:
        """Returns task metadata — used by llm/explainer.py and main.py."""
        return {
            "domain":      "finance",
            "task":        self.task,
            "actions":     self.actions,
            "n_actions":   self.n_actions,
            "state_vars":  self.config["state_vars"],
            "q_shape":     list(self.Q.shape),
            "description": self.config["description"],
            "objectives":  self.config["objectives"],
            "trained_mean":  self.metadata.get("trained_mean",  "not trained yet"),
            "random_mean":   self.metadata.get("random_mean",   "not trained yet"),
            "improvement":   self.metadata.get("improvement_pct", "not trained yet"),
            "data_source":   self.metadata.get("data_source",   "simulation"),
        }

    def __repr__(self):
        return (
            f"FinanceAgent(task='{self.task}', "
            f"n_actions={self.n_actions}, "
            f"q_shape={self.Q.shape})"
        )


# ── Factory function (matches hospital pattern) ───────────────────────────────
def load_finance_agent(task: str) -> FinanceAgent:
    """
    Convenience factory used by main.py and the LLM layer.

    Example:
        agent = load_finance_agent("trading")
        action, q_vals, name = agent.get_action(state)
    """
    return FinanceAgent(task=task)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n" + "="*55)
    print("  FinanceAgent — Self Test")
    print("="*55)

    test_states = {
        "trading": {
            "price_trend":     1,
            "shares_held":     3,
            "cash":            800.0,
            "portfolio_value": 1200.0,
        },
        "savings": {
            "monthly_income":   70.0,
            "current_savings":  250.0,
            "expenses":         45.0,
            "months_remaining": 8.0,
        },
        "budget": {
            "total_budget":          800.0,
            "amount_spent":          300.0,
            "urgent_requests":       2.0,
            "departments_remaining": 3.0,
        },
    }

    weights = {"cost": 0.3, "performance": 0.5, "fairness": 0.2}

    for task, state in test_states.items():
        print(f"\n  Task: {task}")
        print(f"  State: {state}")

        try:
            agent = FinanceAgent(task=task)
            rec   = agent.get_recommendation(state, weights=weights)

            print(f"  Recommended action : {rec['action_name']}")
            print(f"  Confidence         : {rec['confidence']:.4f}")
            print(f"  Q-values           :")
            for a in rec["all_actions"]:
                marker = " ← CHOSEN" if a["chosen"] else ""
                print(f"    {a['name']:<18}: {a['q_value']:+.4f}{marker}")
            print(f"  Agent info         : {agent}")

        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            print(f"  Run training/train_finance.py first.")

    print("\n  Self-test complete.\n")
