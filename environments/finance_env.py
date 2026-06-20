import numpy as np
import random


class FinanceEnv:
    """
    Multi-task Finance simulation following the same pattern as HospitalEnv.

    Three tasks mirror the Finance Domain table:
        "trading"   -- Buy/Sell/Hold trading agent
        "savings"   -- Personal savings management
        "budget"    -- Budget allocation system

    Each episode is one SHIFT_LENGTH of steps. State carries over between
    steps so actions have real downstream consequences.
    """

    SHIFT_LENGTH = 60  # steps per episode

    def __init__(self, task="trading"):
        self.task = task
        self.state = None
        self.done = False
        self.t = 0

        if self.task == "trading":
            self.state_vars = ["price_trend", "shares_held", "cash", "portfolio_value"]
            self.actions = ["Buy", "Sell", "Hold"]
            self.n_actions = 3

        elif self.task == "savings":
            self.state_vars = ["monthly_income", "current_savings", "expenses", "months_remaining"]
            self.actions = ["Save More", "Spend Normal", "Invest"]
            self.n_actions = 3

        elif self.task == "budget":
            self.state_vars = ["total_budget", "amount_spent", "urgent_requests", "departments_remaining"]
            self.actions = ["Allocate Full", "Allocate Partial", "Defer"]
            self.n_actions = 3

        else:
            raise ValueError(f"Unknown task: {task}")

    # ── Reset: start of a new episode ─────────────────────────────────────────
    def reset(self):
        self.t = 0
        self.done = False

        if self.task == "trading":
            self.state = {
                "price_trend":     random.randint(-2, 2),   # -2=crash, 0=flat, 2=bull
                "shares_held":     random.randint(0, 10),
                "cash":            random.randint(200, 800),
                "portfolio_value": random.randint(200, 1000),
            }
            # internal: track share price for P&L
            self._share_price = random.randint(20, 80)

        elif self.task == "savings":
            self.state = {
                "monthly_income":    random.randint(30, 100),  # scaled units
                "current_savings":   random.randint(0, 500),
                "expenses":          random.randint(20, 80),
                "months_remaining":  12,
            }

        elif self.task == "budget":
            self.state = {
                "total_budget":          random.randint(500, 1000),
                "amount_spent":          0,
                "urgent_requests":       random.randint(0, 8),
                "departments_remaining": random.randint(3, 8),
            }

        return self.state

    # ── Step: one decision within the episode ─────────────────────────────────
    def step(self, action_index):
        action = self.actions[action_index]

        if self.task == "trading":
            reward, info = self._step_trading(action)
        elif self.task == "savings":
            reward, info = self._step_savings(action)
        elif self.task == "budget":
            reward, info = self._step_budget(action)

        self.t += 1
        self.done = self.t >= self.SHIFT_LENGTH
        return self.state, reward, self.done, info

    # ── Task 1: Trading Agent ──────────────────────────────────────────────────
    def _step_trading(self, action):
        trend  = self.state["price_trend"]
        shares = self.state["shares_held"]
        cash   = self.state["cash"]
        reward = 0
        info   = {}

        if action == "Buy":
            if cash >= self._share_price:
                self.state["shares_held"] += 1
                self.state["cash"] -= self._share_price
                # good buy in uptrend, risky in downtrend
                if trend >= 1:
                    reward = +8
                    info["result"] = "Smart buy -- uptrend market"
                elif trend <= -1:
                    reward = -6
                    info["result"] = "Risky buy -- downtrend market"
                else:
                    reward = +2
                    info["result"] = "Neutral buy -- flat market"
            else:
                reward = -3
                info["result"] = "Insufficient cash to buy"

        elif action == "Sell":
            if shares > 0:
                self.state["shares_held"] -= 1
                self.state["cash"] += self._share_price
                # good sell in downtrend, missed gains in uptrend
                if trend <= -1:
                    reward = +10
                    info["result"] = "Smart sell -- avoiding downtrend losses"
                elif trend >= 1:
                    reward = -4
                    info["result"] = "Premature sell -- missing uptrend gains"
                else:
                    reward = +3
                    info["result"] = "Neutral sell -- flat market"
            else:
                reward = -3
                info["result"] = "No shares to sell"

        elif action == "Hold":
            # holding in uptrend earns passive gains, holding in downtrend bleeds value
            if trend >= 1:
                reward = +shares * 1.5
                info["result"] = f"Holding {shares} shares -- gaining in uptrend"
            elif trend <= -1:
                reward = -shares * 2
                info["result"] = f"Holding {shares} shares -- losing in downtrend"
            else:
                reward = 0
                info["result"] = "Holding -- flat market, no gain or loss"

        # World dynamics: price moves, trend shifts
        price_move = trend * random.randint(1, 4) + random.randint(-2, 2)
        self._share_price = max(5, min(200, self._share_price + price_move))

        # Trend drifts randomly each step
        trend_shift = random.choices([-1, 0, 0, 0, 1], weights=[15, 40, 20, 15, 10])[0]
        self.state["price_trend"] = max(-2, min(2, trend + trend_shift))

        # Update portfolio value
        self.state["portfolio_value"] = (
            self.state["shares_held"] * self._share_price + self.state["cash"]
        )

        info["share_price"] = self._share_price
        info["trend"]       = self.state["price_trend"]
        return reward, info

    # ── Task 2: Personal Savings Management ───────────────────────────────────
    def _step_savings(self, action):
        income   = self.state["monthly_income"]
        savings  = self.state["current_savings"]
        expenses = self.state["expenses"]
        months   = self.state["months_remaining"]
        reward   = 0
        info     = {}

        if action == "Save More":
            # cut expenses, save aggressively
            saved_this_month = income - (expenses * 0.6)
            if saved_this_month > 0:
                self.state["current_savings"] += saved_this_month
                reward = +10
                info["result"] = f"Saved {saved_this_month:.1f} this month -- great discipline"
            else:
                reward = -5
                info["result"] = "Income too low to save more after expenses"

        elif action == "Spend Normal":
            # balanced -- save what is left after normal spending
            net = income - expenses
            if net > 0:
                self.state["current_savings"] += net * 0.5
                reward = +4
                info["result"] = f"Balanced spending -- saved {net*0.5:.1f}"
            else:
                self.state["current_savings"] = max(0, savings + net)
                reward = -8
                info["result"] = "Overspending -- dipping into savings"

        elif action == "Invest":
            # risky -- invest 30% of savings, chance of gain or loss
            invest_amount = savings * 0.3
            if invest_amount > 0:
                outcome = random.choices(["gain", "loss"], weights=[55, 45])[0]
                if outcome == "gain":
                    gain = invest_amount * random.uniform(0.05, 0.20)
                    self.state["current_savings"] += gain
                    reward = +12
                    info["result"] = f"Investment gained {gain:.1f}"
                else:
                    loss = invest_amount * random.uniform(0.05, 0.15)
                    self.state["current_savings"] = max(0, savings - loss)
                    reward = -6
                    info["result"] = f"Investment lost {loss:.1f}"
            else:
                reward = -2
                info["result"] = "Nothing to invest -- empty savings"

        # World dynamics: income and expenses drift slightly each month
        self.state["months_remaining"] = max(0, months - 1)
        income_drift   = random.randint(-5, 8)
        expenses_drift = random.randint(-3, 5)
        self.state["monthly_income"] = max(10, min(150, income + income_drift))
        self.state["expenses"]       = max(10, min(120, expenses + expenses_drift))

        # Penalty if savings hit zero
        if self.state["current_savings"] <= 0:
            reward -= 10
            info["broke_penalty"] = True

        # Bonus if savings are healthy and months are running out
        if months <= 3 and savings > 200:
            reward += 5
            info["savings_goal_bonus"] = True

        info["savings"] = self.state["current_savings"]
        return reward, info

    # ── Task 3: Budget Allocation System ──────────────────────────────────────
    def _step_budget(self, action):
        total     = self.state["total_budget"]
        spent     = self.state["amount_spent"]
        urgent    = self.state["urgent_requests"]
        depts     = self.state["departments_remaining"]
        remaining = total - spent
        reward    = 0
        info      = {}

        # How much each department requests (random per step)
        request = random.randint(30, 150)

        if action == "Allocate Full":
            if remaining >= request:
                self.state["amount_spent"] += request
                if urgent > 0:
                    reward = +12
                    self.state["urgent_requests"] = max(0, urgent - 1)
                    info["result"] = f"Fully funded urgent request of {request}"
                else:
                    reward = +5
                    info["result"] = f"Fully funded department request of {request}"
            else:
                reward = -8
                info["result"] = "Over budget -- cannot allocate full amount"

        elif action == "Allocate Partial":
            partial = request * random.uniform(0.4, 0.7)
            if remaining >= partial:
                self.state["amount_spent"] += partial
                if urgent > 0:
                    reward = +3
                    info["result"] = f"Partial funding {partial:.0f} for urgent -- not ideal"
                else:
                    reward = +7
                    info["result"] = f"Smart partial allocation of {partial:.0f} -- budget conserved"
            else:
                reward = -5
                info["result"] = "Even partial allocation exceeds remaining budget"

        elif action == "Defer":
            if urgent > 0:
                reward = -12
                info["result"] = "Deferred urgent request -- serious penalty"
            elif depts <= 1:
                reward = -6
                info["result"] = "Deferred last department -- poor planning"
            else:
                reward = +4
                info["result"] = "Deferred non-urgent -- budget preserved strategically"

        # World dynamics: new requests arrive, departments reduce
        new_urgent = random.choices([0, 1, 2], weights=[60, 30, 10])[0]
        self.state["urgent_requests"]       = min(10, urgent + new_urgent)
        self.state["departments_remaining"] = max(0, depts - 1)

        # Penalise going over budget
        if self.state["amount_spent"] > total:
            reward -= 15
            info["over_budget"] = True

        # Bonus for finishing with budget still available and no urgent backlog
        if self.state["departments_remaining"] == 0:
            leftover = total - self.state["amount_spent"]
            if leftover > 0 and self.state["urgent_requests"] == 0:
                reward += 10
                info["clean_finish_bonus"] = leftover

        info["remaining_budget"] = total - self.state["amount_spent"]
        return reward, info

    def get_info(self):
        return {
            "task":       self.task,
            "state_vars": self.state_vars,
            "actions":    self.actions,
        }
