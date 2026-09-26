import numpy as np

from experiment_transition_envs import (
    ExperimentTrafficEnv,
    ExperimentEnergyEnv,
)


def traffic_check():
    env = ExperimentTrafficEnv(task="intersection")

    outcomes = []

    for action in (0, 1):
        np.random.seed(42)
        env.reset(seed=42)

        # Identical queues; switching is allowed.
        env._queues = [[0, 0, 0], [0, 0, 0]]

        # Match accounting to the manually constructed six-vehicle state.
        env._initial_vehicles = sum(len(q) for q in env._queues)
        env._cumulative_arrivals = 0
        env._cumulative_served = 0

        env._sync_state(0, 3)

        np.random.seed(123)
        _, _, _, _, info = env.step(action)

        assert info["executed_action"] == action
        assert info["served_NS"] == (3 if action == 0 else 0)
        assert info["served_EW"] == (3 if action == 1 else 0)

        outcomes.append(list(env._raw_state))

    assert outcomes[0][:2] != outcomes[1][:2]
    print("Traffic: different signals produce different queues — PASS")


def energy_check():
    env = ExperimentEnergyEnv(task="solar_scheduling")
    np.random.seed(42)
    env.reset(seed=42)

    env._raw_state = [8, 3, 4, 1]
    env._state = env._encode_state(env._raw_state)

    _, _, _, _, info = env.step(1)
    assert info["battery_before"] == 4
    assert info["battery_after"] == 6
    assert env._raw_state[2] == 6

    for action in [0, 1, 2] * 10:
        before = env._raw_state[2]
        _, _, _, _, info = env.step(action)

        assert info["battery_before"] == before
        assert env._raw_state[2] == info["battery_after"]

        assert info["battery_after"] == (
            before
            + info["battery_charge"]
            - info["battery_discharge"]
        )
        assert 0 <= info["battery_after"] <= 9

        assert info["demand"] == (
            info["direct_solar"]
            + info["battery_discharge"]
            + info["grid_import"]
        )
        assert info["solar_available"] == (
            info["direct_solar"]
            + info["battery_charge"]
            + info["curtailed_solar"]
        )

    print("Energy: battery persistence and energy balances — PASS")

def traffic_reward_check():
    env = ExperimentTrafficEnv(task="intersection")
    env.reset(seed=42)

    # Ten vehicles wait 2 steps, but only three can depart.
    env._queues = [[2] * 10, []]
    env._sync_state(0, 3)

    # 0.5 * (2 + 2 + 2) + 0.5 * 2 = 4.
    assert np.isclose(env._compute_reward(0), 4.0)

    # Same departing vehicles: extra queued vehicles must not be
    # falsely credited as having been served.
    env._queues = [[2] * 3, []]
    env._sync_state(0, 3)
    assert np.isclose(env._compute_reward(0), 4.0)

    env.close()
    print("Traffic: reward credits actual FIFO service — PASS")

def queue_retention_check():
    env = ExperimentTrafficEnv(task="intersection")
    np.random.seed(42)
    env.reset(seed=42)

    # Force heavy, fixed arrivals to test conservation.
    env._queues = [[], []]
    env._initial_vehicles = 0
    env._cumulative_arrivals = 0
    env._cumulative_served = 0
    env._sync_state(0, 3)
    env._arrivals = lambda: [5, 5]

    for step in range(10):
        _, _, _, _, info = env.step(step % 2)

        assert info["vehicle_balance_error"] == 0
        assert info["overflow_NS"] == 0
        assert info["overflow_EW"] == 0

    physical_total = (
        info["queue_length_NS"] + info["queue_length_EW"]
    )

    assert physical_total > 18
    assert env._raw_state[0] <= 9
    assert env._raw_state[1] <= 9

    print("Traffic: all arrivals retained; observations capped — PASS")


if __name__ == "__main__":
    traffic_check()
    queue_retention_check()
    energy_check()
    traffic_reward_check()