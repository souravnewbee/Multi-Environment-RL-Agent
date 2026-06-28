
import numpy as np
import random
import os



try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False




MARKET_PROFILES = {
    "AAPL":    {"S0": 40.0,    "mu": 0.28,  "sigma": 0.30},   # steady large cap
    "TSLA":    {"S0": 25.0,    "mu": 0.45,  "sigma": 0.75},   # high growth, high vol
    "SPY":     {"S0": 250.0,   "mu": 0.14,  "sigma": 0.18},   # broad market index
    "MSFT":    {"S0": 100.0,   "mu": 0.30,  "sigma": 0.28},   # stable tech
    "BTC-USD": {"S0": 4000.0,  "mu": 0.60,  "sigma": 1.20},   # crypto extreme vol
}


REGIME_OVERLAYS = {
    "AAPL": [
        (280, 310, -0.025),   # crash period: -2.5%/day
        (310, 400, +0.018),   # recovery: +1.8%/day
    ],
    "TSLA": [
        (280, 310, -0.040),
        (310, 450, +0.030),
        (700, 780, -0.025),   # 2022 correction
    ],
    "SPY": [
        (280, 305, -0.030),
        (305, 400, +0.020),
        (680, 760, -0.012),
    ],
    "MSFT": [
        (280, 310, -0.022),
        (310, 400, +0.016),
    ],
    "BTC-USD": [
        (200, 240, -0.060),
        (240, 450, +0.045),
        (650, 750, -0.040),
    ],
}

TRADING_DAYS_PER_YEAR = 252


def generate_synthetic_prices(ticker="AAPL", n_days=1260):
    """
    Generate synthetic price series using Geometric Brownian Motion.
    n_days=1260 = 5 years of trading days (252/year).
    """
    profile = MARKET_PROFILES.get(ticker, MARKET_PROFILES["AAPL"])
    S0      = profile["S0"]
    mu      = profile["mu"]
    sigma   = profile["sigma"]
    dt      = 1.0 / TRADING_DAYS_PER_YEAR

    prices = [S0]
    for day in range(1, n_days):
        Z      = np.random.normal(0, 1)
        shock  = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        S_next = prices[-1] * np.exp(shock)

        # Apply regime overlay if applicable
        for (start, end, daily_return) in REGIME_OVERLAYS.get(ticker, []):
            if start <= day < end:
                S_next = prices[-1] * (1 + daily_return + np.random.normal(0, 0.01))

        prices.append(max(0.01, S_next))

    return np.array(prices)


def load_real_prices(ticker="AAPL"):
    """Try to load real prices from yfinance, return None if unavailable."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.download(ticker,
                         start="2019-01-01",
                         end="2024-01-01",
                         progress=False,
                         auto_adjust=True)
        if df.empty or len(df) < 100:
            return None
        return df["Close"].values.flatten().astype(float)
    except Exception:
        return None


def compute_trend(prices, idx, window=5):
    """Compute price trend as integer -2 to +2 from recent window."""
    if idx < window:
        return 0
    recent    = prices[idx - window: idx]
    trend_raw = (recent[-1] - recent[0]) / max(recent[0], 0.01)
    return int(np.clip(trend_raw * 20, -2, 2))


def trend_label(trend):
    return {-2: "CRASH", -1: "DOWN", 0: "FLAT", 1: "UP", 2: "BULL"}.get(trend, "FLAT")


# ═════════════════════════════════════════════════════════════════════════════
class FinanceEnv:
    """
    Multi-task Finance Environment.
    Identical structural pattern to HospitalEnv.
    """

    SHIFT_LENGTH = 60   # steps per episode

    def __init__(self, task="trading", ticker="AAPL", use_real_data=True):
        self.task          = task
        self.ticker        = ticker
        self.use_real_data = use_real_data
        self.state         = None
        self.done          = False
        self.t             = 0

        # ── Define state/action spaces per task ──────────────────────────────
        if self.task == "trading":
            self.state_vars = [
                "price_trend",       # -2 to +2
                "shares_held",       # 0 to 20
                "cash",              # dollars
                "portfolio_value",   # total worth
            ]
            self.actions   = ["Buy", "Sell", "Hold"]
            self.n_actions = 3
            self._load_price_data()

        elif self.task == "savings":
            self.state_vars = [
                "monthly_income",    # scaled units
                "current_savings",
                "expenses",
                "months_remaining",
            ]
            self.actions   = ["Save More", "Spend Normal", "Invest"]
            self.n_actions = 3

        elif self.task == "budget":
            self.state_vars = [
                "total_budget",
                "amount_spent",
                "urgent_requests",
                "departments_remaining",
            ]
            self.actions   = ["Allocate Full", "Allocate Partial", "Defer"]
            self.n_actions = 3

        else:
            raise ValueError(f"Unknown task: {task}. "
                             f"Choose from: trading, savings, budget")

    # ── Price data loading ────────────────────────────────────────────────────
    def _load_price_data(self):
        self._prices = None

        if self.use_real_data:
            self._prices = load_real_prices(self.ticker)
            if self._prices is not None:
                self._data_source = f"real yfinance ({self.ticker})"
                return

        # Fallback: generate synthetic GBM prices
        np.random.seed(42)   # reproducible synthetic data
        self._prices      = generate_synthetic_prices(self.ticker, n_days=1260)
        self._data_source = f"synthetic GBM ({self.ticker})"

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        self.t    = 0
        self.done = False

        if self.task == "trading":
            # Pick random start, leave room for full episode
            max_start         = len(self._prices) - self.SHIFT_LENGTH - 10
            self._episode_start = random.randint(5, max(6, max_start))
            self._current_idx   = self._episode_start

            self._share_price = float(self._prices[self._current_idx])
            trend             = compute_trend(self._prices, self._current_idx)

            self.state = {
                "price_trend":     trend,
                "shares_held":     random.randint(0, 5),
                "cash":            float(random.randint(500, 2000)),
                "portfolio_value": 0.0,
            }
            self.state["portfolio_value"] = (
                self.state["shares_held"] * self._share_price
                + self.state["cash"]
            )

        elif self.task == "savings":
            self.state = {
                "monthly_income":   float(random.randint(30, 100)),
                "current_savings":  float(random.randint(0,  500)),
                "expenses":         float(random.randint(20,  80)),
                "months_remaining": 12.0,
            }

        elif self.task == "budget":
            self.state = {
                "total_budget":          float(random.randint(500, 1000)),
                "amount_spent":          0.0,
                "urgent_requests":       float(random.randint(0, 8)),
                "departments_remaining": float(random.randint(3, 8)),
            }

        return self.state

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action_index):
        if action_index < 0 or action_index >= self.n_actions:
            raise ValueError(f"Invalid action {action_index}. "
                             f"Must be 0 to {self.n_actions-1}.")

        action = self.actions[action_index]

        if self.task == "trading":
            reward, info = self._step_trading(action)
        elif self.task == "savings":
            reward, info = self._step_savings(action)
        elif self.task == "budget":
            reward, info = self._step_budget(action)

        self.t   += 1
        self.done = self.t >= self.SHIFT_LENGTH

        return self.state, reward, self.done, info

    # ══════════════════════════════════════════════════════════════════════════
    # Task 1 — Trading (with real/synthetic price data)
    # ══════════════════════════════════════════════════════════════════════════
    def _step_trading(self, action):
        trend  = self.state["price_trend"]
        shares = self.state["shares_held"]
        cash   = self.state["cash"]
        reward = 0.0
        info   = {}

        # ── Multi-objective reward components ─────────────────────────────────
        # cost component:        -1 per share bought (transaction cost proxy)
        # performance component: profit/loss based on action + trend alignment
        # fairness component:    not deeply applicable to trading; kept neutral

        if action == "Buy":
            if cash >= self._share_price:
                self.state["shares_held"] += 1
                self.state["cash"]        -= self._share_price

                if trend == 2:
                    r_perf = +12.0
                    info["result"] = "Excellent buy -- strong bull market"
                elif trend == 1:
                    r_perf = +8.0
                    info["result"] = "Good buy -- uptrend market"
                elif trend == 0:
                    r_perf = +2.0
                    info["result"] = "Neutral buy -- flat market"
                elif trend == -1:
                    r_perf = -6.0
                    info["result"] = "Risky buy -- downtrend market"
                else:   # trend == -2
                    r_perf = -12.0
                    info["result"] = "Dangerous buy -- market crashing"

                r_cost     = -1.0   # transaction cost
                r_fairness = 0.0
            else:
                reward = -3.0
                info["result"] = f"Cannot buy -- need ${self._share_price:.2f}, have ${cash:.2f}"
                return self._advance_price(reward, info)

        elif action == "Sell":
            if shares > 0:
                self.state["shares_held"] -= 1
                self.state["cash"]        += self._share_price

                if trend == -2:
                    r_perf = +12.0
                    info["result"] = "Excellent sell -- avoiding crash"
                elif trend == -1:
                    r_perf = +8.0
                    info["result"] = "Good sell -- avoiding downtrend"
                elif trend == 0:
                    r_perf = +3.0
                    info["result"] = "Neutral sell -- flat market"
                elif trend == 1:
                    r_perf = -5.0
                    info["result"] = "Premature sell -- missing uptrend gains"
                else:   # trend == 2
                    r_perf = -10.0
                    info["result"] = "Very premature sell -- missing bull run"

                r_cost     = -0.5   # smaller transaction cost on sell
                r_fairness = 0.0
            else:
                reward = -3.0
                info["result"] = "No shares to sell"
                return self._advance_price(reward, info)

        elif action == "Hold":
            if trend >= 1:
                r_perf = shares * (1.5 * trend)   # scales with trend strength
                info["result"] = f"Holding {shares} shares -- gaining in {trend_label(trend)}"
            elif trend <= -1:
                r_perf = shares * (2.0 * trend)   # negative, scales with crash severity
                info["result"] = f"Holding {shares} shares -- losing in {trend_label(trend)}"
            else:
                r_perf = 0.0
                info["result"] = f"Holding {shares} shares -- flat market"

            r_cost     = 0.0
            r_fairness = 0.0

        # Combined reward (weights applied in training script via env wrapper)
        reward = r_perf + r_cost
        info["r_performance"] = r_perf
        info["r_cost"]        = r_cost
        info["r_fairness"]    = r_fairness

        return self._advance_price(reward, info)

    def _advance_price(self, reward, info):
        """Move to next real/synthetic trading day and update state."""
        self._current_idx += 1

        # Guard against running off end of price array
        if self._current_idx >= len(self._prices):
            self._current_idx = len(self._prices) - 1
            self.done = True

        self._share_price = float(self._prices[self._current_idx])
        new_trend = compute_trend(self._prices, self._current_idx)

        self.state["price_trend"]     = new_trend
        self.state["portfolio_value"] = (
            self.state["shares_held"] * self._share_price
            + self.state["cash"]
        )

        info["share_price"]     = round(self._share_price, 2)
        info["trend"]           = new_trend
        info["trend_label"]     = trend_label(new_trend)
        info["portfolio_value"] = round(self.state["portfolio_value"], 2)
        info["data_source"]     = self._data_source
        return reward, info

    # ══════════════════════════════════════════════════════════════════════════
    # Task 2 — Savings Management
    # ══════════════════════════════════════════════════════════════════════════
    def _step_savings(self, action):
        income   = self.state["monthly_income"]
        savings  = self.state["current_savings"]
        expenses = self.state["expenses"]
        months   = self.state["months_remaining"]
        reward   = 0.0
        info     = {}

        if action == "Save More":
            # Reduce discretionary spending to 60% of normal
            saved = income - (expenses * 0.6)
            if saved > 0:
                self.state["current_savings"] += saved
                r_perf     = +10.0
                r_cost     = -2.0   # quality of life cost
                r_fairness = +1.0
                info["result"]       = f"Saved {saved:.1f} this month -- disciplined"
                info["amount_saved"] = saved
            else:
                r_perf = r_cost = r_fairness = 0.0
                reward = -5.0
                info["result"] = "Income too low to save more after reduced expenses"
                return self._finalise_savings(reward, info)

        elif action == "Spend Normal":
            net = income - expenses
            if net >= 0:
                self.state["current_savings"] += net * 0.5
                r_perf     = +4.0
                r_cost     = 0.0
                r_fairness = +2.0
                info["result"]       = f"Balanced month -- saved {net*0.5:.1f}"
                info["amount_saved"] = net * 0.5
            else:
                drawdown = abs(net)
                self.state["current_savings"] = max(0.0, savings - drawdown)
                r_perf     = -8.0
                r_cost     = -5.0
                r_fairness = -1.0
                info["result"]    = f"Overspending by {drawdown:.1f} -- drawing from savings"
                info["drawdown"]  = drawdown

        elif action == "Invest":
            invest_amount = savings * 0.30
            if invest_amount > 10:
                # Outcome probabilities calibrated to approximate stock market
                outcome = random.choices(
                    ["big_gain", "small_gain", "flat", "small_loss", "big_loss"],
                    weights=[15, 30, 20, 25, 10]
                )[0]
                outcomes = {
                    "big_gain":   (savings + invest_amount * random.uniform(0.15, 0.30),  +15, -2),
                    "small_gain": (savings + invest_amount * random.uniform(0.03, 0.15),  +10, -1),
                    "flat":       (savings,                                                 +1,  0),
                    "small_loss": (max(0, savings - invest_amount * random.uniform(0.03, 0.12)), -5, -1),
                    "big_loss":   (max(0, savings - invest_amount * random.uniform(0.12, 0.25)), -10, -2),
                }
                new_sav, r_perf, r_cost = outcomes[outcome]
                self.state["current_savings"] = new_sav
                r_fairness = 0.0
                change = new_sav - savings
                info["result"]         = f"Investment {outcome} -- savings changed by {change:+.1f}"
                info["invested"]       = invest_amount
                info["outcome"]        = outcome
            else:
                reward = -2.0
                info["result"] = "Savings too low to invest meaningfully"
                return self._finalise_savings(reward, info)

        # Combined reward
        reward = r_perf + r_cost + r_fairness

        # Hard constraint penalties
        if self.state["current_savings"] <= 0:
            reward -= 15.0
            info["broke_penalty"] = True
            info["result"] += " | SAVINGS EMPTY -- emergency penalty"

        # Bonus for finishing strong
        if months <= 1 and self.state["current_savings"] >= 300:
            reward += 10.0
            info["goal_bonus"] = True

        info["r_performance"] = r_perf
        info["r_cost"]        = r_cost
        info["r_fairness"]    = r_fairness

        return self._finalise_savings(reward, info)

    def _finalise_savings(self, reward, info):
        """Apply world dynamics for savings task."""
        income   = self.state["monthly_income"]
        expenses = self.state["expenses"]
        months   = self.state["months_remaining"]

        # Income and expenses drift each month
        income_shock   = random.choices(
            [-10, -5, 0, 0, 5, 10, 15],
            weights=[5, 10, 25, 25, 20, 10, 5]
        )[0]
        expense_shock  = random.choices(
            [-5, 0, 0, 5, 10],
            weights=[10, 30, 25, 25, 10]
        )[0]

        self.state["monthly_income"]   = float(max(10, min(150, income + income_shock)))
        self.state["expenses"]         = float(max(10, min(120, expenses + expense_shock)))
        self.state["months_remaining"] = float(max(0, months - 1))

        info["savings"]  = round(self.state["current_savings"], 1)
        info["income"]   = self.state["monthly_income"]
        info["expenses"] = self.state["expenses"]
        return reward, info

    # ══════════════════════════════════════════════════════════════════════════
    # Task 3 — Budget Allocation
    # ══════════════════════════════════════════════════════════════════════════
    def _step_budget(self, action):
        total   = self.state["total_budget"]
        spent   = self.state["amount_spent"]
        urgent  = self.state["urgent_requests"]
        depts   = self.state["departments_remaining"]
        remaining = total - spent
        reward  = 0.0
        info    = {}

        # Each step a department submits a request
        request = float(random.randint(30, 150))
        info["request_size"] = request

        if action == "Allocate Full":
            if remaining >= request:
                self.state["amount_spent"] += request
                if urgent > 0:
                    r_perf     = +15.0
                    r_fairness = +5.0
                    r_cost     = -3.0
                    self.state["urgent_requests"] = max(0.0, urgent - 1)
                    info["result"] = f"Fully funded urgent request (${request:.0f})"
                else:
                    r_perf     = +6.0
                    r_fairness = +2.0
                    r_cost     = -1.0
                    info["result"] = f"Fully funded department (${request:.0f})"
            else:
                r_perf = r_fairness = 0.0
                r_cost = 0.0
                reward = -10.0
                info["result"] = f"Over budget -- cannot allocate ${request:.0f} (only ${remaining:.0f} left)"
                return self._finalise_budget(reward, info)

        elif action == "Allocate Partial":
            ratio   = random.uniform(0.40, 0.70)
            partial = request * ratio
            if remaining >= partial:
                self.state["amount_spent"] += partial
                if urgent > 0:
                    r_perf     = +5.0
                    r_fairness = -2.0   # urgent request not fully met
                    r_cost     = +3.0   # budget conserved
                    info["result"] = f"Partially funded urgent request (${partial:.0f} of ${request:.0f})"
                else:
                    r_perf     = +4.0
                    r_fairness = +1.0
                    r_cost     = +5.0
                    info["result"] = f"Smart partial allocation (${partial:.0f} of ${request:.0f})"
            else:
                r_perf = r_fairness = r_cost = 0.0
                reward = -6.0
                info["result"] = f"Even partial amount exceeds remaining budget"
                return self._finalise_budget(reward, info)

        elif action == "Defer":
            if urgent > 0:
                r_perf     = -15.0
                r_fairness = -8.0
                r_cost     = +5.0
                info["result"] = f"Deferred URGENT request -- serious fairness penalty"
            elif depts <= 1:
                r_perf     = -8.0
                r_fairness = -3.0
                r_cost     = +3.0
                info["result"] = "Deferred last department -- poor planning"
            else:
                r_perf     = -2.0
                r_fairness = +0.0
                r_cost     = +6.0
                info["result"] = "Deferred non-urgent request -- budget conserved"

        # Combined reward
        reward = r_perf + r_fairness + r_cost

        # Over budget hard penalty
        if self.state["amount_spent"] > total * 1.05:
            reward -= 20.0
            info["over_budget_penalty"] = True

        # Clean finish bonus
        if depts <= 1 and remaining > 0 and urgent == 0:
            reward += 8.0
            info["clean_finish_bonus"] = remaining

        info["r_performance"] = r_perf
        info["r_cost"]        = r_cost
        info["r_fairness"]    = r_fairness

        return self._finalise_budget(reward, info)

    def _finalise_budget(self, reward, info):
        """Apply world dynamics for budget task."""
        urgent = self.state["urgent_requests"]
        depts  = self.state["departments_remaining"]

        # New urgent requests can appear each step
        new_urgent = random.choices([0, 1, 2], weights=[60, 30, 10])[0]
        self.state["urgent_requests"]       = float(min(10.0, urgent + new_urgent))
        self.state["departments_remaining"] = float(max(0.0, depts - 1))

        info["remaining_budget"]    = round(self.state["total_budget"] - self.state["amount_spent"], 1)
        info["urgent_remaining"]    = self.state["urgent_requests"]
        info["departments_left"]    = self.state["departments_remaining"]
        return reward, info

    # ── Utility ───────────────────────────────────────────────────────────────
    def get_info(self):
        base = {
            "task":       self.task,
            "state_vars": self.state_vars,
            "actions":    self.actions,
        }
        if self.task == "trading":
            base["ticker"]      = self.ticker
            base["data_source"] = self._data_source
            base["price_days"]  = len(self._prices)
        return base

    def render(self):
        """Print current state to terminal."""
        print(f"\n  [FinanceEnv | task={self.task} | step={self.t}/{self.SHIFT_LENGTH}]")
        for k, v in self.state.items():
            if isinstance(v, float):
                print(f"    {k:<25}: {v:.2f}")
            else:
                print(f"    {k:<25}: {v}")
        if self.task == "trading":
            print(f"    {'share_price':<25}: {self._share_price:.2f}")
            print(f"    {'data_source':<25}: {self._data_source}")