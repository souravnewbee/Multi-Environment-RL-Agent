"""
environments/agriculture_env.py
================================
Multi-task Agriculture Environment for UMORDA.
Gymnasium-compatible. Follows the exact same pattern as HospitalEnv.

NEW: Real-time weather integration via SQLAlchemy + Open-Meteo.
     When real weather data is available (run weather_fetcher.py first),
     the environment uses actual Bangladesh weather patterns instead of
     random numbers. Falls back to simulation if DB is not available.

Three tasks:
    "soil_preparation" -- Soil preparation for exotic fruit cultivation
    "irrigation"       -- Smart irrigation using real rainfall/weather data
    "pest_control"     -- Treatment resource allocation across farm plots

Run weather_fetcher.py first to populate the weather database:
    python data/weather_fetcher.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import sys
from datetime import date, timedelta

# ── Try importing weather DB (optional -- falls back to simulation) ───────────
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from data.weather_fetcher import WeatherFetcher
    WEATHER_DB_AVAILABLE = True
except ImportError:
    WEATHER_DB_AVAILABLE = False


class AgricultureEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    SHIFT_LENGTH = 40   # steps per episode (days in active management phase)

    def __init__(self, task="soil_preparation", render_mode=None,
                 use_real_weather=True):
        super().__init__()
        self.task            = task
        self.render_mode     = render_mode
        self.use_real_weather = use_real_weather and WEATHER_DB_AVAILABLE
        self.state           = None
        self.t               = 0
        self._weather_cache  = []   # loaded once per episode from DB
        self._weather_idx    = 0    # current position in weather cache

        # ── Load weather DB if available ──────────────────────────────────────
        self._fetcher = None
        if self.use_real_weather:
            try:
                self._fetcher = WeatherFetcher()
                n = self._fetcher.total_records()
                if n < 30:
                    print(f"  [AgricultureEnv] Only {n} weather records in DB.")
                    print(f"  Run: python data/weather_fetcher.py")
                    self.use_real_weather = False
                else:
                    print(f"  [AgricultureEnv] Real weather DB loaded ({n} records).")
            except Exception as e:
                print(f"  [AgricultureEnv] Weather DB unavailable: {e}")
                self.use_real_weather = False

        if not self.use_real_weather:
            print("  [AgricultureEnv] Using synthetic weather simulation.")

        # ── Define state/action spaces per task ──────────────────────────────
        if self.task == "soil_preparation":
            self.state_vars = [
                "soil_ph",
                "organic_matter",
                "drainage_quality",
                "days_remaining",
            ]
            self.actions   = ["Add Compost", "Adjust pH", "Improve Drainage", "Plant Now"]
            self.n_actions = 4
            self.PH_TARGET_LOW,  self.PH_TARGET_HIGH = 5.5, 7.0
            self.OM_TARGET       = 60
            self.DRAINAGE_TARGET = 60
            self.PLANTING_WINDOW = 25
            self.observation_space = spaces.Box(
                low  = np.array([3.0,  0,  0,  0], dtype=np.float32),
                high = np.array([9.0, 100, 100, self.PLANTING_WINDOW], dtype=np.float32),
            )

        elif self.task == "irrigation":
            self.state_vars = [
                "water_reservoir",
                "crop_stress",
                "rainfall_trend",   # -2 drought → +2 heavy rain (from real DB or synthetic)
                "days_remaining",
            ]
            self.actions   = ["Irrigate Heavy", "Irrigate Light", "Skip Irrigation"]
            self.n_actions = 3
            self.observation_space = spaces.Box(
                low  = np.array([0,  0, -2, 0], dtype=np.float32),
                high = np.array([100, 100, 2, self.SHIFT_LENGTH], dtype=np.float32),
            )

        elif self.task == "pest_control":
            self.state_vars = [
                "total_resource",
                "resource_used",
                "urgent_outbreaks",
                "plots_remaining",
            ]
            self.actions   = ["Full Treatment", "Partial Treatment", "Defer"]
            self.n_actions = 3
            self.observation_space = spaces.Box(
                low  = np.array([0,    0,  0,  0], dtype=np.float32),
                high = np.array([1000, 1000, 10, 10], dtype=np.float32),
            )

        else:
            raise ValueError(f"Unknown task: {task}")

        self.action_space = spaces.Discrete(self.n_actions)

    # ── Weather loading ───────────────────────────────────────────────────────
    def _load_weather_sequence(self):
        """
        Load a sequence of SHIFT_LENGTH weather records from DB.
        Picks a random start date to vary episodes across seasons.
        Falls back to synthetic generation if DB is unavailable.
        """
        if not self.use_real_weather or not self._fetcher:
            return self._synthetic_weather_sequence()

        try:
            # Pick random start in last 2 years, leave room for full episode
            today      = date.today()
            max_offset = 700
            offset     = random.randint(self.SHIFT_LENGTH, max_offset)
            end_date   = today - timedelta(days=offset)
            start_date = end_date - timedelta(days=self.SHIFT_LENGTH)

            records = self._fetcher.get_date_range(start_date, end_date)

            if len(records) < 10:
                return self._synthetic_weather_sequence()

            # Pad with synthetic if DB records < SHIFT_LENGTH
            while len(records) < self.SHIFT_LENGTH:
                records.append(self._synthetic_day())

            return records[:self.SHIFT_LENGTH]

        except Exception:
            return self._synthetic_weather_sequence()

    def _synthetic_weather_sequence(self):
        """Generate synthetic weather for one episode."""
        # Pick a random season to make episodes varied
        season = random.choice(["monsoon", "dry", "pre_monsoon", "winter"])
        season_params = {
            "monsoon":     {"rain_mu": 15.0, "rain_sigma": 12.0, "trend_bias":  1},
            "dry":         {"rain_mu":  1.0, "rain_sigma":  2.0, "trend_bias": -1},
            "pre_monsoon": {"rain_mu":  5.0, "rain_sigma":  6.0, "trend_bias":  0},
            "winter":      {"rain_mu":  0.5, "rain_sigma":  1.0, "trend_bias": -1},
        }
        p     = season_params[season]
        seq   = []
        trend = p["trend_bias"]
        for _ in range(self.SHIFT_LENGTH):
            rain = max(0, np.random.normal(p["rain_mu"], p["rain_sigma"]))
            et0  = random.uniform(3, 7)
            surplus = rain - et0
            if surplus >= 15:   trend = min( 2, trend + 1)
            elif surplus >= 5:  trend = min( 2, trend)
            elif surplus >= -3: trend = max(-2, trend)
            elif surplus >= -10:trend = max(-2, trend - 1)
            else:               trend = max(-2, trend - 1)
            seq.append({
                "rain":           round(rain, 2),
                "temp_mean":      round(random.uniform(18, 38), 1),
                "rainfall_trend": int(trend),
                "drought_index":  round(surplus, 2),
                "is_monsoon":     random.choice([True, False]),
                "evapotranspiration": round(et0, 2),
                "source":         "synthetic",
            })
        return seq

    def _synthetic_day(self):
        rain = max(0, np.random.normal(5.0, 8.0))
        et0  = random.uniform(3, 7)
        return {
            "rain":           round(rain, 2),
            "temp_mean":      round(random.uniform(20, 36), 1),
            "rainfall_trend": self._rain_to_trend(rain, et0),
            "drought_index":  round(rain - et0, 2),
            "is_monsoon":     False,
            "evapotranspiration": round(et0, 2),
            "source":         "synthetic_fallback",
        }

    @staticmethod
    def _rain_to_trend(rain, et0):
        surplus = rain - et0
        if surplus >= 15:  return  2
        if surplus >= 5:   return  1
        if surplus >= -3:  return  0
        if surplus >= -10: return -1
        return -2

    def _current_weather(self) -> dict:
        """Return current day's weather from cache."""
        if self._weather_cache and self._weather_idx < len(self._weather_cache):
            return self._weather_cache[self._weather_idx]
        return self._synthetic_day()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        # Load a fresh weather sequence for this episode
        if self.task in ("irrigation", "soil_preparation"):
            self._weather_cache = self._load_weather_sequence()
            self._weather_idx   = 0

        if self.task == "soil_preparation":
            self.state = {
                "soil_ph":          round(random.uniform(4.0, 8.0), 2),
                "organic_matter":   float(random.randint(10, 40)),
                "drainage_quality": float(random.randint(10, 40)),
                "days_remaining":   float(self.PLANTING_WINDOW),
            }

        elif self.task == "irrigation":
            w = self._current_weather()
            self.state = {
                "water_reservoir": float(random.randint(40, 100)),
                "crop_stress":     float(random.randint(0, 30)),
                # Use real rainfall trend if available, else synthetic
                "rainfall_trend":  float(w.get("rainfall_trend", random.choice([-2,-1,0,1,2]))),
                "days_remaining":  float(self.SHIFT_LENGTH),
            }

        elif self.task == "pest_control":
            self.state = {
                "total_resource":   float(random.randint(500, 1000)),
                "resource_used":    0.0,
                "urgent_outbreaks": float(random.randint(0, 5)),
                "plots_remaining":  float(random.randint(4, 10)),
            }

        return self._obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action_index):
        action     = self.actions[action_index]
        terminated = False

        if self.task == "soil_preparation":
            reward, terminated, info = self._step_soil_preparation(action)
        elif self.task == "irrigation":
            reward, info = self._step_irrigation(action)
        elif self.task == "pest_control":
            reward, info = self._step_pest_control(action)

        self.t += 1
        self._weather_idx += 1   # advance weather timeline

        if self.task == "soil_preparation":
            self.state["days_remaining"] = max(0, self.PLANTING_WINDOW - self.t)
            truncated = (self.t >= self.PLANTING_WINDOW) and not terminated
            if truncated:
                reward -= 20.0
                info["missed_window"] = True
                info["result"] = info.get("result","") + " | Planting window closed!"
        elif self.task == "irrigation":
            self.state["days_remaining"] = max(0, self.SHIFT_LENGTH - self.t)
            truncated = self.t >= self.SHIFT_LENGTH
        else:
            truncated = self.t >= self.SHIFT_LENGTH

        info["step"]         = self.t
        info["weather_source"] = (
            "real_db" if self.use_real_weather else "synthetic"
        )

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════════
    # Task 1 — Soil Preparation (unchanged logic, now gets real temp context)
    # ══════════════════════════════════════════════════════════════════════════
    def _step_soil_preparation(self, action):
        ph       = self.state["soil_ph"]
        om       = self.state["organic_matter"]
        drainage = self.state["drainage_quality"]
        terminated = False
        info = {}

        # Real weather context for soil preparation
        w = self._current_weather()
        temp     = w.get("temp_mean", 28.0)
        rain     = w.get("rain", 5.0)
        info["weather"] = {"temp": temp, "rain": round(rain, 1)}

        if action == "Add Compost":
            # Compost effectiveness slightly reduced in extreme heat
            heat_factor = 1.0 if temp < 35 else 0.7
            gain = random.uniform(8, 15) * heat_factor
            self.state["organic_matter"] = min(100, om + gain)
            self.state["soil_ph"]        = max(3.0, ph - random.uniform(0, 0.15))
            r_perf = 4.0 if om < self.OM_TARGET else 1.0
            r_cost = -2.0
            r_fair = 0.0
            info["result"] = (f"Added compost -- organic matter now "
                              f"{self.state['organic_matter']:.1f}%"
                              f"{' (heat-reduced)' if heat_factor < 1 else ''}")

        elif action == "Adjust pH":
            if ph < self.PH_TARGET_LOW:
                self.state["soil_ph"] = min(9.0, ph + random.uniform(0.3, 0.6))
            elif ph > self.PH_TARGET_HIGH:
                self.state["soil_ph"] = max(3.0, ph - random.uniform(0.3, 0.6))
            else:
                self.state["soil_ph"] = ph + random.uniform(-0.1, 0.1)
            in_range = self.PH_TARGET_LOW <= self.state["soil_ph"] <= self.PH_TARGET_HIGH
            r_perf = 5.0 if in_range else 1.0
            r_cost = -2.0
            r_fair = 0.0
            info["result"] = f"Adjusted pH -- now {self.state['soil_ph']:.2f}"

        elif action == "Improve Drainage":
            # Heavy rain days make drainage improvement more urgent (+bonus)
            rain_bonus = 2.0 if rain > 20 else 0.0
            gain = random.uniform(8, 15)
            self.state["drainage_quality"] = min(100, drainage + gain)
            r_perf = 4.0 + rain_bonus if drainage < self.DRAINAGE_TARGET else 1.0
            r_cost = -3.0
            r_fair = 0.0
            info["result"] = (f"Improved drainage -- now "
                              f"{self.state['drainage_quality']:.1f}%"
                              f"{' (+rain bonus)' if rain_bonus > 0 else ''}")

        elif action == "Plant Now":
            ph_ok    = self.PH_TARGET_LOW <= self.state["soil_ph"] <= self.PH_TARGET_HIGH
            om_ok    = self.state["organic_matter"]   >= self.OM_TARGET
            drain_ok = self.state["drainage_quality"] >= self.DRAINAGE_TARGET
            checks   = sum([ph_ok, om_ok, drain_ok])

            # Monsoon season penalty -- planting in heavy rain is risky
            monsoon_penalty = -5.0 if w.get("is_monsoon") and rain > 15 else 0.0

            if checks == 3:
                r_perf = 30.0 + monsoon_penalty
                info["result"] = "Planted -- soil fully ready, excellent conditions"
            elif checks == 2:
                r_perf = 8.0 + monsoon_penalty
                info["result"] = "Planted -- soil mostly ready"
            elif checks == 1:
                r_perf = -10.0
                info["result"] = "Planted -- soil poorly prepared, high risk of failure"
            else:
                r_perf = -25.0
                info["result"] = "Planted -- soil completely unprepared"

            if monsoon_penalty < 0:
                info["result"] += f" | Monsoon penalty ({monsoon_penalty:.0f})"

            r_cost = 0.0
            r_fair = 0.0
            terminated = True
            info["checks_passed"] = checks
            info["ph_ok"]         = ph_ok
            info["om_ok"]         = om_ok
            info["drain_ok"]      = drain_ok

        reward = r_perf + r_cost + r_fair
        info["r_performance"] = r_perf
        info["r_cost"]        = r_cost
        info["r_fairness"]    = r_fair
        return reward, terminated, info

    # ══════════════════════════════════════════════════════════════════════════
    # Task 2 — Irrigation (now uses REAL weather data from DB)
    # ══════════════════════════════════════════════════════════════════════════
    def _step_irrigation(self, action):
        reservoir = self.state["water_reservoir"]
        stress    = self.state["crop_stress"]
        trend     = self.state["rainfall_trend"]
        info      = {}

        # Pull real weather for today
        w         = self._current_weather()
        real_rain = w.get("rain", 0.0)
        real_temp = w.get("temp_mean", 28.0)
        real_et0  = w.get("evapotranspiration", 5.0)
        is_monsoon = w.get("is_monsoon", False)
        source     = w.get("source", "synthetic")

        info["weather"] = {
            "rain_mm":   round(real_rain, 1),
            "temp_c":    real_temp,
            "et0_mm":    round(real_et0, 1),
            "is_monsoon": is_monsoon,
            "source":    source,
        }

        if action == "Irrigate Heavy":
            use = 15
            if reservoir >= use:
                self.state["water_reservoir"] -= use
                self.state["crop_stress"]      = max(0, stress - 20)
                r_perf = 8.0 if stress > 40 else 2.0
                r_cost = -4.0
                # Penalise heavy irrigation during monsoon -- wasteful
                if is_monsoon and real_rain > 10:
                    r_cost -= 3.0
                    info["monsoon_waste"] = True
                info["result"] = "Heavy irrigation -- crop stress relieved"
            else:
                r_perf, r_cost = -5.0, 0.0
                info["result"] = "Insufficient reservoir for heavy irrigation"

        elif action == "Irrigate Light":
            use = 6
            if reservoir >= use:
                self.state["water_reservoir"] -= use
                self.state["crop_stress"]      = max(0, stress - 8)
                r_perf = 4.0 if stress > 20 else 1.0
                r_cost = -1.5
                info["result"] = "Light irrigation applied"
            else:
                r_perf, r_cost = -3.0, 0.0
                info["result"] = "Insufficient reservoir for light irrigation"

        elif action == "Skip Irrigation":
            r_cost = +2.0
            if trend <= -1:   # drought conditions
                self.state["crop_stress"] = min(100, stress + 15)
                r_perf = -6.0
                info["result"] = "Skipped during drought -- crop stress rising"
            else:
                self.state["crop_stress"] = min(100, stress + 3)
                r_perf = 1.0
                info["result"] = "Skipped -- conditions mild"

        r_fair = 0.0

        # ── REAL weather dynamics ─────────────────────────────────────────────
        # Reservoir refills based on ACTUAL rainfall from DB
        reservoir_gain = real_rain * 0.7   # 70% capture efficiency
        self.state["water_reservoir"] = min(
            100, self.state["water_reservoir"] + reservoir_gain
        )

        # Crop stress increases from heat (high ET0 means more plant stress)
        heat_stress = max(0, real_et0 - 6.0) * 0.5   # extra stress above 6mm ET0
        self.state["crop_stress"] = min(100, self.state["crop_stress"] + heat_stress)

        # Update rainfall trend from REAL next day's data
        next_w = self._weather_cache[min(self._weather_idx + 1,
                                         len(self._weather_cache) - 1)]
        self.state["rainfall_trend"] = float(
            next_w.get("rainfall_trend", trend)
        )

        # Critical stress penalty
        if self.state["crop_stress"] >= 90:
            r_perf -= 10.0
            info["critical_stress"] = True

        reward = r_perf + r_cost + r_fair
        info["r_performance"]  = r_perf
        info["r_cost"]         = r_cost
        info["r_fairness"]     = r_fair
        info["reservoir_gain"] = round(reservoir_gain, 1)
        info["heat_stress"]    = round(heat_stress, 2)
        return reward, info

    # ══════════════════════════════════════════════════════════════════════════
    # Task 3 — Pest Control (unchanged -- pest dynamics are internal)
    # ══════════════════════════════════════════════════════════════════════════
    def _step_pest_control(self, action):
        total     = self.state["total_resource"]
        used      = self.state["resource_used"]
        urgent    = self.state["urgent_outbreaks"]
        plots     = self.state["plots_remaining"]
        remaining = total - used
        info      = {}

        request = float(random.randint(40, 150))
        info["request_size"] = request

        if action == "Full Treatment":
            if remaining >= request:
                self.state["resource_used"] += request
                if urgent > 0:
                    r_perf, r_fair, r_cost = 15.0, 5.0, -3.0
                    self.state["urgent_outbreaks"] = max(0, urgent - 1)
                    info["result"] = f"Fully treated urgent outbreak (${request:.0f})"
                else:
                    r_perf, r_fair, r_cost = 6.0, 2.0, -1.0
                    info["result"] = f"Fully treated plot (${request:.0f})"
            else:
                reward = -10.0
                info["result"] = (f"Over budget -- need ${request:.0f}, "
                                  f"have ${remaining:.0f}")
                info["r_performance"] = info["r_cost"] = info["r_fairness"] = 0.0
                return self._finalise_pest(reward, info)

        elif action == "Partial Treatment":
            ratio   = random.uniform(0.4, 0.7)
            partial = request * ratio
            if remaining >= partial:
                self.state["resource_used"] += partial
                if urgent > 0:
                    r_perf, r_fair, r_cost = 5.0, -2.0, 3.0
                    info["result"] = f"Partially treated urgent (${partial:.0f}/${request:.0f})"
                else:
                    r_perf, r_fair, r_cost = 4.0, 1.0, 5.0
                    info["result"] = f"Smart partial treatment (${partial:.0f}/${request:.0f})"
            else:
                reward = -6.0
                info["result"] = "Even partial exceeds remaining resources"
                info["r_performance"] = info["r_cost"] = info["r_fairness"] = 0.0
                return self._finalise_pest(reward, info)

        elif action == "Defer":
            if urgent > 0:
                r_perf, r_fair, r_cost = -15.0, -8.0, 5.0
                info["result"] = "Deferred URGENT outbreak -- crop damage risk"
            elif plots <= 1:
                r_perf, r_fair, r_cost = -8.0, -3.0, 3.0
                info["result"] = "Deferred last plot -- poor planning"
            else:
                r_perf, r_fair, r_cost = -2.0, 0.0, 6.0
                info["result"] = "Deferred non-urgent -- resources conserved"

        reward = r_perf + r_fair + r_cost

        if self.state["resource_used"] > total * 1.05:
            reward -= 20.0
            info["over_budget_penalty"] = True

        if plots <= 1 and remaining > 0 and urgent == 0:
            reward += 8.0
            info["clean_finish_bonus"] = remaining

        info["r_performance"] = r_perf
        info["r_cost"]        = r_cost
        info["r_fairness"]    = r_fair
        return self._finalise_pest(reward, info)

    def _finalise_pest(self, reward, info):
        urgent = self.state["urgent_outbreaks"]
        plots  = self.state["plots_remaining"]
        new_u  = random.choices([0, 1, 2], weights=[60, 30, 10])[0]
        self.state["urgent_outbreaks"] = min(10, urgent + new_u)
        self.state["plots_remaining"]  = max(0, plots - 1)
        info["remaining_resource"] = round(
            self.state["total_resource"] - self.state["resource_used"], 1
        )
        info["urgent_remaining"] = self.state["urgent_outbreaks"]
        info["plots_left"]       = self.state["plots_remaining"]
        return reward, info

    # ── Observation array ─────────────────────────────────────────────────────
    def _obs(self):
        if self.task == "soil_preparation":
            return np.array([
                self.state["soil_ph"],
                self.state["organic_matter"],
                self.state["drainage_quality"],
                self.state["days_remaining"],
            ], dtype=np.float32)
        elif self.task == "irrigation":
            return np.array([
                self.state["water_reservoir"],
                self.state["crop_stress"],
                self.state["rainfall_trend"],
                self.state["days_remaining"],
            ], dtype=np.float32)
        elif self.task == "pest_control":
            return np.array([
                self.state["total_resource"],
                self.state["resource_used"],
                self.state["urgent_outbreaks"],
                self.state["plots_remaining"],
            ], dtype=np.float32)

    def render(self):
        w_info = ""
        if self.use_real_weather and self._weather_cache:
            w = self._current_weather()
            w_info = (f" | rain={w.get('rain',0):.1f}mm "
                      f"temp={w.get('temp_mean',28):.1f}°C "
                      f"trend={w.get('rainfall_trend',0)}")
        print(f"  [t={self.t:02d}] {self.task}{w_info} | {self.state}")

    def get_info(self):
        return {
            "domain":         "agriculture",
            "task":           self.task,
            "state_vars":     self.state_vars,
            "actions":        self.actions,
            "n_actions":      self.n_actions,
            "shift_length":   self.SHIFT_LENGTH,
            "weather_source": "real_db" if self.use_real_weather else "synthetic",
        }