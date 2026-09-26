

import argparse
import csv
import inspect
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np


TASKS = {
    "er_queue": "Hospital",
    "intersection": "Traffic",
    "solar_scheduling": "Energy",
    "budget": "Finance",
    "pest_control": "Agriculture",
}

METRIC_NAMES = {
    "er_queue": "Successful-service opportunity rate",
    "intersection": "Mean served-vehicle waiting time (steps)",
    "solar_scheduling": "Grid-import fraction of demand",
    "budget": "Urgent full-funding opportunity rate",
    "pest_control": "Urgent full-treatment opportunity rate",
}


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))


def make_env(task):
    if task == "er_queue":
        from environments.hospital_env import HospitalEnv
        return HospitalEnv(task=task)

    if task == "intersection":
        from experiment_transition_envs import ExperimentTrafficEnv
        return ExperimentTrafficEnv(task=task)

    if task == "solar_scheduling":
        from experiment_transition_envs import ExperimentEnergyEnv
        return ExperimentEnergyEnv(task=task)

    if task == "budget":
        from environments.finance_env import FinanceEnv
        return FinanceEnv(task=task)

    from environments.agriculture_env import AgricultureEnv
    return AgricultureEnv(task=task)


def reset_env(env, seed):
    # Current environments use both Python random and global NumPy RNG.
    seed_all(seed)
    if "seed" in inspect.signature(env.reset).parameters:
        result = env.reset(seed=int(seed))
    else:
        result = env.reset()

    return result[0] if isinstance(result, tuple) else result


def snapshot(env):
    if hasattr(env, "_raw_state"):
        return list(env._raw_state)
    return dict(env.state)


def state_index(task, obs):
    if task in ("intersection", "solar_scheduling"):
        return int(obs)

    if task == "er_queue":
        from agents.hospital_agent import discretize
        return discretize(obs, task)

    if task == "pest_control":
        from agents.agriculture_agent import discretize
        return discretize(obs, task)

    # Matches FinanceAgent._discretize() for budget.
    total = max(float(obs["total_budget"]), 1.0)
    return (
        int(np.clip(int(float(obs["amount_spent"]) / total * 5), 0, 4)),
        int(np.clip(int(obs["urgent_requests"]), 0, 4)),
        int(np.clip(int(obs["departments_remaining"]), 0, 4)),
    )


def q_shape(task, env):
    if task in ("intersection", "solar_scheduling"):
        return (env.observation_space.n, env.action_space.n)

    if task == "er_queue":
        from agents.hospital_agent import Q_SHAPES
        return Q_SHAPES[task]

    if task == "pest_control":
        from agents.agriculture_agent import Q_SHAPES
        return Q_SHAPES[task]

    return (5, 5, 5, 3)


def rule_action(task, state, env):
    """Fixed rules; no thresholds fitted to evaluation results."""
    if task == "er_queue":
        return 0 if state["emergency_queue"] > 0 else 1

    if task == "intersection":
        ns, ew, phase, elapsed, wait_ns, wait_ew = state
        limit = env.cfg["max_wait_limit"]

        # Uses the same capped observation available to Q-learning.
        if max(wait_ns, wait_ew) >= limit:
            if wait_ns == wait_ew:
                return 1 - int(phase)
            return 0 if wait_ns > wait_ew else 1

        if elapsed < env.cfg["min_phase_duration"]:
            return int(phase)

        return 0 if ns >= ew else 1

    if task == "solar_scheduling":
        solar, demand, battery, _ = state

        if solar > 0 and demand > 0:
            return 0  # Use Solar Directly

        if solar > 0 and battery < 9:
            return 1  # Store in Battery

        return 2      # Use Grid

    if task == "budget":
        total, used = "total_budget", "amount_spent"
        urgent, count = "urgent_requests", "departments_remaining"
    else:
        total, used = "total_resource", "resource_used"
        urgent, count = "urgent_outbreaks", "plots_remaining"

    remaining = state[total] - state[used]

    # Request size is drawn inside step(), so it is unavailable to BOTH
    # policies before selection. Maximum request is 150; maximum partial
    # allocation is 150 * 0.7 = 105 in these environments.
    if state[urgent] > 0:
        return 0 if remaining >= 150 else 1 if remaining >= 105 else 2

    allowance = remaining / max(1, state[count])
    return 0 if allowance >= 150 else 1 if allowance >= 105 else 2


def take_step(env, action, reward_mode):
    result = env.step(int(action))

    if len(result) == 5:
        obs, native_reward, terminated, truncated, info = result
    else:
        obs, native_reward, terminated, info = result
        truncated = False

    reward = float(native_reward)

    if reward_mode == "equal-components":
        keys = ("r_performance", "r_cost", "r_fairness")
        if all(key in info for key in keys):
            components = [float(info[key]) for key in keys]

            # Preserve penalties/bonuses not included in component fields.
            residual = reward - sum(components)
            reward = sum(components) / 3.0 + residual

    return obs, reward, bool(terminated), bool(truncated), info


def train(task, seed, args, folder):
    seed_all(seed)
    env = make_env(task)

    # Exploration has a separate RNG so environment draws cannot perturb it.
    exploration_rng = np.random.default_rng(seed)
    q = exploration_rng.uniform(-0.01, 0.01, q_shape(task, env))
    epsilon = 1.0

    for episode in range(args.train_episodes):
        obs = reset_env(env, seed * 100000 + episode)

        for step in range(args.max_steps):
            state = state_index(task, obs)

            if exploration_rng.random() < epsilon:
                action = int(exploration_rng.integers(q.shape[-1]))
            else:
                action = int(np.argmax(q[state]))

            nxt, reward, terminated, truncated, _ = take_step(
                env, action, args.reward_mode
            )

            # Treat the capped evaluation/training episode as a finite
            # episode, consistently with the existing hospital trainer.
            reached_cap = step + 1 >= args.max_steps
            done = terminated or truncated or reached_cap
            target = reward if done else (
                reward + args.gamma * np.max(q[state_index(task, nxt)])
            )

            q[state][action] += args.alpha * (
                target - q[state][action]
            )

            obs = nxt
            if done:
                break

        epsilon = max(0.01, epsilon * args.epsilon_decay)

        if (episode + 1) % 1000 == 0:
            print(
                f"{task}, seed {seed}: "
                f"{episode + 1}/{args.train_episodes}",
                flush=True,
            )

    np.save(folder / f"{task}_seed_{seed}.npy", q)
    return q


def jain(values):
    values = np.asarray(values, dtype=float)
    denominator = len(values) * np.square(values).sum()

    # All-zero vectors have undefined Jain fairness, rather than 1 by fiat.
    if denominator == 0:
        return None

    return float(values.sum() ** 2 / denominator)


def evaluate(task, q, policy, training_seed, args):
    env = make_env(task)
    rows = []

    for episode in range(args.eval_episodes):
        evaluation_seed = args.eval_seed + episode
        obs = reset_env(env, evaluation_seed)

        total_reward = 0.0
        starvation = 0

        opportunities = 0.0
        successes = 0.0

        burdens = np.zeros(2)
        unserved_streaks = np.zeros(2, dtype=int)
        fulfillment_ratios = []

        traffic_served = 0
        traffic_served_wait = 0.0
        traffic_arrivals = 0
        traffic_overflow = 0
        traffic_remaining = 0

        constraint_overrides = 0
        wait_limit_overrides = 0
        minimum_phase_overrides = 0

        energy_demand = 0.0
        energy_grid_import = 0.0
        energy_solar_available = 0.0
        energy_direct_solar = 0.0
        energy_battery_charge = 0.0
        energy_battery_discharge = 0.0
        energy_curtailed_solar = 0.0

        action_counts = {}

        for step in range(args.max_steps):
            before = snapshot(env)

            proposed = (
                rule_action(task, before, env)
                if policy == "rule"
                else int(np.argmax(q[state_index(task, obs)]))
            )

            step_seed = np.random.SeedSequence(
                [evaluation_seed, step]
            ).generate_state(1)[0]
            seed_all(step_seed)

            nxt, reward, terminated, truncated, info = take_step(
                env, proposed, args.reward_mode
            )

            actual = int(info.get("executed_action", proposed))
            action_counts[actual] = action_counts.get(actual, 0) + 1

            total_reward += reward
            after = snapshot(env)

            if task == "er_queue":
                counts = [
                    before["emergency_queue"],
                    before["normal_queue"],
                ]

                for group, count in enumerate(counts):
                    waiting = count > 0
                    served = waiting and actual == group

                    if served or not waiting:
                        unserved_streaks[group] = 0
                    else:
                        unserved_streaks[group] += 1

                    starvation += int(
                        unserved_streaks[group] == args.er_threshold
                    )

                    # Queue burden measured after service, before arrivals.
                    burdens[group] += max(0, count - int(served))

                # An opportunity exists when either queue contains a patient.
                # A success means the selected queue was nonempty.
                opportunities += int(any(count > 0 for count in counts))
                successes += int(counts[actual] > 0)

            elif task == "intersection":
                required = (
                    "served_NS",
                    "served_EW",
                    "served_wait_sum_NS",
                    "served_wait_sum_EW",
                    "arrivals_NS",
                    "arrivals_EW",
                    "overflow_NS",
                    "overflow_EW",
                    "queue_wait_sum_NS",
                    "queue_wait_sum_EW",
                    "true_max_wait_NS",
                    "true_max_wait_EW",
                    "constraint_reason",
                )

                missing = [key for key in required if key not in info]
                if missing:
                    raise RuntimeError(
                        "Traffic evaluation requires the revised "
                        f"ExperimentTrafficEnv. Missing: {missing}"
                    )

                traffic_served += (
                    info["served_NS"] + info["served_EW"]
                )
                traffic_served_wait += (
                    info["served_wait_sum_NS"]
                    + info["served_wait_sum_EW"]
                )
                traffic_arrivals += (
                    info["arrivals_NS"] + info["arrivals_EW"]
                )
                traffic_overflow += (
                    info["overflow_NS"] + info["overflow_EW"]
                )

                # Retain exact waiting burden, not capped observation waits.
                burdens += [
                    info["queue_wait_sum_NS"],
                    info["queue_wait_sum_EW"],
                ]

                limit = env.cfg["max_wait_limit"]

                starvation += int(
                    info["queue_length_NS"] > 0
                    and info["true_max_wait_NS"] >= limit
                )
                starvation += int(
                    info["queue_length_EW"] > 0
                    and info["true_max_wait_EW"] >= limit
                )

                reason = info["constraint_reason"]
                constraint_overrides += int(actual != proposed)
                wait_limit_overrides += int(reason == "wait_limit")
                minimum_phase_overrides += int(
                    reason == "minimum_phase_duration"
                )

                traffic_remaining = int(
                    info["queue_length_NS"] + info["queue_length_EW"]
                )

            elif task in ("budget", "pest_control"):
                if task == "budget":
                    used = "amount_spent"
                    urgent = "urgent_requests"
                else:
                    used = "resource_used"
                    urgent = "urgent_outbreaks"

                allocated = max(0.0, after[used] - before[used])
                request = float(info["request_size"])

                fraction = min(1.0, allocated / request)
                fulfillment_ratios.append(fraction)

                is_urgent = before[urgent] > 0
                fully_met = fraction >= 1.0 - 1e-8

                opportunities += int(is_urgent)
                successes += int(is_urgent and fully_met)

                # Urgent opportunities not fully met:
                # includes partial allocation, deferral and failed allocation.
                starvation += int(is_urgent and not fully_met)

            elif task == "solar_scheduling":
                required = (
                    "demand",
                    "demand_served",
                    "grid_import",
                    "solar_available",
                    "direct_solar",
                    "battery_charge",
                    "battery_discharge",
                    "curtailed_solar",
                )

                missing = [key for key in required if key not in info]
                if missing:
                    raise RuntimeError(
                        "Energy evaluation requires the revised "
                        f"ExperimentEnergyEnv. Missing: {missing}"
                    )

                energy_demand += info["demand"]
                energy_grid_import += info["grid_import"]
                energy_solar_available += info["solar_available"]
                energy_direct_solar += info["direct_solar"]
                energy_battery_charge += info["battery_charge"]
                energy_battery_discharge += info["battery_discharge"]
                energy_curtailed_solar += info["curtailed_solar"]

                # Actual unmet-demand steps; expected to be zero with
                # unlimited grid backup.
                starvation += int(
                    info["demand_served"] < info["demand"] - 1e-8
                )

            obs = nxt
            if terminated or truncated:
                break

        steps = step + 1

        if task == "intersection":
            key_metric = (
                traffic_served_wait / traffic_served
                if traffic_served > 0 else None
            )
            fairness = jain(burdens)

        elif task == "solar_scheduling":
            key_metric = (
                energy_grid_import / energy_demand
                if energy_demand > 0 else None
            )
            fairness = None

        elif task in ("budget", "pest_control"):
            key_metric = (
                successes / opportunities
                if opportunities > 0 else None
            )
            fairness = jain(fulfillment_ratios)

        else:
            key_metric = (
                successes / opportunities
                if opportunities > 0 else None
            )
            fairness = jain(burdens)

        rows.append({
            "domain": TASKS[task],
            "task": task,
            "policy": policy,
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
            "steps": steps,
            "reward": total_reward,
            "starvation": starvation,
            "jain_fairness": fairness,
            "key_metric": key_metric,

            "constraint_overrides": constraint_overrides,
            "wait_limit_overrides": wait_limit_overrides,
            "minimum_phase_overrides": minimum_phase_overrides,

            "traffic_vehicles_served": traffic_served,
            "traffic_served_wait_sum": traffic_served_wait,
            "traffic_arrivals": traffic_arrivals,
            "traffic_overflow": traffic_overflow,
            "traffic_remaining_at_end": traffic_remaining,

            "energy_demand": energy_demand,
            "energy_grid_import": energy_grid_import,
            "energy_solar_available": energy_solar_available,
            "energy_direct_solar": energy_direct_solar,
            "energy_battery_charge": energy_battery_charge,
            "energy_battery_discharge": energy_battery_discharge,
            "energy_curtailed_solar": energy_curtailed_solar,

            "executed_action_counts": json.dumps(action_counts),
        })

    return rows


def average(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return float(np.mean(values)) if values else None


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def formatted(value):
    return "N/A" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 2026])
    parser.add_argument("--eval-seed", type=int, default=9000000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--er-threshold", type=int, default=8)
    parser.add_argument("--reward-mode",
                        choices=["native", "equal-components"],
                        default="native")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS),
                        default=list(TASKS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least three distinct training seeds.")

    if min(args.train_episodes, args.eval_episodes,
           args.max_steps, args.er_threshold) <= 0:
        parser.error("Episode counts, thresholds and step cap must be positive.")

    folder = args.output or Path(
        "experiment_results"
    ) / datetime.now().strftime("professor_%Y%m%d_%H%M%S")

    # Prevent accidental replacement of previous experiments.
    folder.mkdir(parents=True, exist_ok=False)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["numpy_version"] = np.__version__
    config["key_metrics"] = METRIC_NAMES
    (folder / "configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    raw_rows = []
    seed_rows = []
    table_v = []
    table_vi = []

    for task in args.tasks:
        baseline = evaluate(task, None, "rule", "", args)
        raw_rows.extend(baseline)
        all_q_rows = []
        task_seed_rows = []

        for seed in args.seeds:
            q = train(task, seed, args, folder)
            episodes = evaluate(task, q, "q_learning", seed, args)
            raw_rows.extend(episodes)
            all_q_rows.extend(episodes)

            summary = {
                "domain": TASKS[task],
                "task": task,
                "seed": seed,
            }
            for field in ("reward", "starvation",
                          "jain_fairness", "key_metric"):
                summary[field] = average(episodes, field)
                summary[field + "_valid_episodes"] = sum(
                    row[field] is not None for row in episodes
                )

            seed_rows.append(summary)
            task_seed_rows.append(summary)

            # Save completed work after each seed.
            write_csv(folder / "episodes.csv", raw_rows)
            write_csv(folder / "seed_summary.csv", seed_rows)

        table_v.append({
            "domain": TASKS[task],
            "task": task,
            "q_reward": average(all_q_rows, "reward"),
            "rule_reward": average(baseline, "reward"),
            "metric_name": METRIC_NAMES[task],
            "q_key_metric": average(all_q_rows, "key_metric"),
            "rule_key_metric": average(baseline, "key_metric"),
        })

        robustness = {"domain": TASKS[task], "task": task}
        for field in ("reward", "starvation", "jain_fairness"):
            values = [
                row[field] for row in task_seed_rows
                if row[field] is not None
            ]
            robustness[field + "_mean"] = (
                float(np.mean(values)) if values else None
            )
            robustness[field + "_sd"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1 else None
            )

        table_vi.append(robustness)
        write_csv(folder / "table_v.csv", table_v)
        write_csv(folder / "table_vi.csv", table_vi)

    lines = [
        "| Domain | Task | Q reward | Rule reward | Metric | Q metric | Rule metric |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in table_v:
        cells = [
            row["domain"], row["task"],
            formatted(row["q_reward"]),
            formatted(row["rule_reward"]),
            row["metric_name"],
            formatted(row["q_key_metric"]),
            formatted(row["rule_key_metric"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "| Domain | Task | Reward mean ± SD | Starvation mean ± SD | Jain mean ± SD |",
        "|---|---|---:|---:|---:|",
    ])
    for row in table_vi:
        cells = [row["domain"], row["task"]]
        for field in ("reward", "starvation", "jain_fairness"):
            mean = row[field + "_mean"]
            sd = row[field + "_sd"]
            cells.append(
                "N/A" if mean is None
                else f"{formatted(mean)} ± {formatted(sd)}"
            )
        lines.append("| " + " | ".join(cells) + " |")

    (folder / "tables.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nResults saved to: {folder.resolve()}")


if __name__ == "__main__":
    main()