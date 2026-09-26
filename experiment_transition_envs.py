"""
Revised transition models for the professor experiments.

Traffic assumptions:
- CSV directional counts are simulated arrivals per step.
- FIFO queues, capacity 9 vehicles per direction.
- Green serves up to 3 vehicles per step.
- Overflow is recorded, never silently counted as served.
- Wait observations remain capped at 9 for existing Q-table encoding.

Solar assumptions:
- Existing 0–9 scales are abstract energy units, not kWh.
- Battery capacity 9; charge/discharge limit 2 units per step.
- Ideal conversion efficiency.
- Unlimited grid backup supplies remaining household demand.
- Existing reward functions are retained.

These assumptions must be documented if used in the paper.
"""

import numpy as np

from environments.traffic_env import TrafficEnv
from environments.energy_env import EnergyEnv


class ExperimentTrafficEnv(TrafficEnv):
    SERVICE_CAPACITY = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if self.task != "intersection":
            return self._state, {}

        ns, ew, phase, elapsed, _, _ = self._raw_state

        self._queues = [
            [0] * int(ns),
            [0] * int(ew),
        ]
        self._initial_vehicles = int(ns + ew)
        self._cumulative_arrivals = 0
        self._cumulative_served = 0

        self._sync_state(int(phase), int(elapsed))

        return self._state, {
            "transition_model": "unbounded_fifo_queues",
            "initial_vehicles": self._initial_vehicles,
        }

    def _sync_state(self, phase, elapsed):
        """Cap observations, never physical queues or actual waiting ages."""
        waits = [
            max(queue, default=0)
            for queue in self._queues
        ]

        self._raw_state = [
            min(9, len(self._queues[0])),
            min(9, len(self._queues[1])),
            int(phase),
            min(9, int(elapsed)),
            min(9, int(waits[0])),
            min(9, int(waits[1])),
        ]

        self._state = self._encode_state(self._raw_state)

    def _safety_override(self, action):
        if self.task != "intersection":
            return super()._safety_override(action)

        phase = int(self._raw_state[2])
        elapsed = int(self._raw_state[3])
        limit = self.cfg["max_wait_limit"]

        # Use actual waits, not capped observations.
        waits = [
            max(queue, default=0)
            for queue in self._queues
        ]
        urgent = [
            bool(self._queues[d]) and waits[d] >= limit
            for d in range(2)
        ]

        if urgent[0] or urgent[1]:
            if urgent[0] and urgent[1]:
                if waits[0] == waits[1]:
                    return 1 - phase
                return 0 if waits[0] > waits[1] else 1

            return 0 if urgent[0] else 1

        if elapsed < self.cfg["min_phase_duration"]:
            return phase

        return int(action)

    def _arrivals(self):
        if self._using_real_data:
            self._data_idx += 1
            row = self._cityflow_data[
                self._data_idx % len(self._cityflow_data)
            ]

            # CSV pattern counts are interpreted as arrivals per step.
            return [
                max(0, int(row[0])),
                max(0, int(row[1])),
            ]

        return [
            int(np.random.randint(0, 3)),
            int(np.random.randint(0, 3)),
        ]

    def _compute_reward(self, action):
        """Reward actual FIFO service using uncapped physical waiting times."""
        if self.task != "intersection":
            return super()._compute_reward(action)

        action = int(action)
        selected = self._queues[action]
        other = self._queues[1 - action]

        # Must match the vehicles removed by step(): oldest vehicles first.
        departing = sorted(selected, reverse=True)[:self.SERVICE_CAPACITY]

        served_wait_sum = float(sum(departing))
        selected_max_wait = float(max(selected, default=0))
        other_max_wait = float(max(other, default=0))

        phase = int(self._raw_state[2])
        elapsed = int(self._raw_state[3])
        minimum_duration = int(self.cfg["min_phase_duration"])
        wait_limit = int(self.cfg["max_wait_limit"])

        # Preserve the original coefficients, but use physical service.
        reward = 0.5 * served_wait_sum
        reward -= 0.3 * other_max_wait

        # Award urgency only if at least one vehicle is actually served.
        if departing:
            reward += 0.5 * selected_max_wait

        if other and other_max_wait >= wait_limit - 2:
            reward -= 8.0

        if action != phase and elapsed < minimum_duration:
            reward -= 5.0

        # Preserve the original reward range.
        return float(np.clip(reward, -20.0, 20.0))

    def step(self, action):
        if self.task != "intersection":
            return super().step(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid traffic action: {action}")

        proposed = int(action)
        previous_phase = int(self._raw_state[2])
        previous_elapsed = int(self._raw_state[3])

        actual_waits_before = [
            max(queue, default=0)
            for queue in self._queues
        ]

        executed = self._safety_override(proposed)
        reason = None

        if executed != proposed:
            limit = self.cfg["max_wait_limit"]
            wait_limit_reached = any(
                self._queues[d]
                and actual_waits_before[d] >= limit
                for d in range(2)
            )

            reason = (
                "wait_limit"
                if wait_limit_reached
                else "minimum_phase_duration"
            )

        # Preserve the existing reward for this transition revision.
        # It still uses capped state features; evaluate actual outcomes too.
        reward = float(self._compute_reward(executed))

        served = [0, 0]
        served_wait_sum = [0, 0]

        queue = self._queues[executed]
        count = min(self.SERVICE_CAPACITY, len(queue))
        departed = queue[:count]
        self._queues[executed] = queue[count:]

        served[executed] = count
        served_wait_sum[executed] = sum(departed)

        # Existing unserved vehicles wait one more simulation step.
        for direction in range(2):
            self._queues[direction] = [
                age + 1 for age in self._queues[direction]
            ]

        arrivals = self._arrivals()

        # Retain every arrival. There is no queue-capacity rejection.
        for direction in range(2):
            self._queues[direction].extend(
                [0] * arrivals[direction]
            )

        self._cumulative_arrivals += sum(arrivals)
        self._cumulative_served += sum(served)

        elapsed = (
            1 if executed != previous_phase
            else previous_elapsed + 1
        )
        self._sync_state(executed, elapsed)
        self._step_count += 1

        lengths = [
            len(self._queues[0]),
            len(self._queues[1]),
        ]
        waits = [
            max(self._queues[0], default=0),
            max(self._queues[1], default=0),
        ]

        # Check that no vehicles disappear or are created by service.
        assert (
            self._initial_vehicles + self._cumulative_arrivals
            == self._cumulative_served + sum(lengths)
        ), "Traffic vehicle conservation failed"

        info = {
            "proposed_action": proposed,
            "executed_action": executed,
            "constraint_override": executed != proposed,
            "constraint_reason": reason,
            "safety_override": reason == "wait_limit",

            "arrivals_NS": arrivals[0],
            "arrivals_EW": arrivals[1],
            "served_NS": served[0],
            "served_EW": served[1],
            "served_wait_sum_NS": served_wait_sum[0],
            "served_wait_sum_EW": served_wait_sum[1],

            # Kept for compatibility with the existing evaluator.
            "overflow_NS": 0,
            "overflow_EW": 0,

            "queue_length_NS": lengths[0],
            "queue_length_EW": lengths[1],
            "queue_wait_sum_NS": sum(self._queues[0]),
            "queue_wait_sum_EW": sum(self._queues[1]),
            "true_max_wait_NS": waits[0],
            "true_max_wait_EW": waits[1],

            "initial_vehicles": self._initial_vehicles,
            "cumulative_arrivals": self._cumulative_arrivals,
            "cumulative_served": self._cumulative_served,
            "vehicle_balance_error": (
                self._initial_vehicles
                + self._cumulative_arrivals
                - self._cumulative_served
                - sum(lengths)
            ),
        }

        if self.render_mode == "human":
            self.render()

        return (
            self._state,
            reward,
            False,
            self._step_count >= self.max_steps,
            info,
        )

class ExperimentEnergyEnv(EnergyEnv):
    BATTERY_CAPACITY = 9
    CHARGE_LIMIT = 2
    DISCHARGE_LIMIT = 2

    def _next_solar_inputs(self):
        """Advance external solar and demand without replacing battery."""
        if self._using_real_data:
            self._data_idx += 1
            row = self._solar_rows[
                self._data_idx % len(self._solar_rows)
            ]

            solar = int(np.clip(row[0], 0, 9))
            time_of_day = int(np.clip(row[1], 0, 3))
        else:
            # Four simulation steps per time category.
            time_of_day = (
                int(self._raw_state[3])
                + int((self._step_count + 1) % 4 == 0)
            ) % 4

            solar_ranges = {
                0: (1, 6),   # morning
                1: (4, 10),  # afternoon
                2: (0, 3),   # evening
                3: (0, 1),   # night
            }
            low, high = solar_ranges[time_of_day]
            solar = int(np.random.randint(low, high))

        base_demand = {0: 4, 1: 3, 2: 6, 3: 2}
        demand = int(np.clip(
            base_demand[time_of_day] + np.random.randint(-1, 2),
            0,
            9,
        ))

        return solar, demand, time_of_day

    def step(self, action):
        if self.task != "solar_scheduling":
            return super().step(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid energy action: {action}")

        solar, demand, battery_before, time_of_day = map(
            int, self._raw_state
        )
        battery_after = battery_before

        direct_solar = 0
        battery_discharge = 0
        battery_charge = 0
        grid_import = 0

        if action == 0:  # Use Solar Directly, with backup
            direct_solar = min(solar, demand)
            remaining_demand = demand - direct_solar

            battery_discharge = min(
                remaining_demand,
                battery_before,
                self.DISCHARGE_LIMIT,
            )
            battery_after -= battery_discharge
            grid_import = remaining_demand - battery_discharge

        elif action == 1:  # Store solar; grid supplies the home
            battery_charge = min(
                solar,
                self.BATTERY_CAPACITY - battery_before,
                self.CHARGE_LIMIT,
            )
            battery_after += battery_charge
            grid_import = demand

        else:  # Buy from Grid; battery remains unchanged
            grid_import = demand

        curtailed_solar = solar - direct_solar - battery_charge

        # Retain the existing reward for now. Its alignment with measured
        # energy outcomes must be assessed before the final experiment.
        reward = float(self._compute_reward(int(action)))

        next_solar, next_demand, next_time = self._next_solar_inputs()
        self._raw_state = [
            next_solar,
            next_demand,
            battery_after,
            next_time,
        ]
        self._state = self._encode_state(self._raw_state)
        self._step_count += 1

        info = {
            "demand": demand,
            "solar_available": solar,
            "direct_solar": direct_solar,
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge,
            "battery_before": battery_before,
            "battery_after": battery_after,
            "grid_import": grid_import,
            "curtailed_solar": curtailed_solar,
            "demand_served": (
                direct_solar + battery_discharge + grid_import
            ),
            "time_of_day": time_of_day,
        }

        return (
            self._state,
            reward,
            False,
            self._step_count >= self.max_steps,
            info,
        )