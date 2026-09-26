"""
Audit the saved professor experiment without retraining.

Run from the repository root:
    python audit_experiment_e.py --results experiment_results/professor_20260927_000940
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import experiment_e as experiment


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(values)) if values else None


def action_names(task, env):
    if hasattr(env, "cfg"):
        return env.cfg["action_meanings"]
    return env.actions


def mode_description(env):
    flag = getattr(env, "_using_real_data", None)
    if flag is None:
        return "synthetic"
    return "csv" if flag else "synthetic"


def audit_policy(task, policy, seed, q, args, output):
    env = experiment.make_env(task)
    names = action_names(task, env)

    actions = Counter()
    overrides = 0
    empty_emergency_actions = 0
    normal_service = 0
    zero_remaining_steps = 0
    failed_allocations = 0
    feasible_but_deferred = 0
    native_rewards = []
    selected_rewards = []
    episode_lengths = []

    step_rows = []

    for episode in range(args.eval_episodes):
        evaluation_seed = args.eval_seed + episode
        obs = experiment.reset_env(env, evaluation_seed)
        native_total = 0.0
        selected_total = 0.0

        for step in range(args.max_steps):
            before = experiment.snapshot(env)

            if policy == "rule":
                proposed = experiment.rule_action(task, before, env)
            else:
                proposed = int(
                    np.argmax(q[experiment.state_index(task, obs)])
                )

            step_seed = np.random.SeedSequence(
                [evaluation_seed, step]
            ).generate_state(1)[0]
            experiment.seed_all(step_seed)

            actual = (
                env._safety_override(proposed)
                if task == "intersection"
                else proposed
            )

            # Step once. Compute both reward versions from that same transition.
            result = env.step(int(proposed))
            if len(result) == 5:
                nxt, native, terminated, truncated, info = result
            else:
                nxt, native, terminated, info = result
                truncated = False

            native = float(native)
            selected = native

            keys = ("r_performance", "r_cost", "r_fairness")
            components_available = all(key in info for key in keys)

            if args.reward_mode == "equal-components" and components_available:
                component_sum = sum(float(info[key]) for key in keys)
                selected = component_sum / 3.0 + native - component_sum

            after = experiment.snapshot(env)
            native_total += native
            selected_total += selected
            actions[names[actual]] += 1
            overrides += int(actual != proposed)

            empty_er = False
            served_normal = False
            zero_remaining = False
            failed = False
            deferred_feasible = False
            request = None
            allocated = None

            if task == "er_queue":
                empty_er = (
                    actual == 0 and before["emergency_queue"] == 0
                )
                served_normal = (
                    actual == 1 and before["normal_queue"] > 0
                )
                empty_emergency_actions += int(empty_er)
                normal_service += int(served_normal)

            elif task in ("budget", "pest_control"):
                if task == "budget":
                    total_key = "total_budget"
                    used_key = "amount_spent"
                    count_key = "departments_remaining"
                else:
                    total_key = "total_resource"
                    used_key = "resource_used"
                    count_key = "plots_remaining"

                zero_remaining = before[count_key] <= 0
                zero_remaining_steps += int(zero_remaining)

                request = float(info["request_size"])
                allocated = max(0.0, after[used_key] - before[used_key])
                remaining = before[total_key] - before[used_key]

                failed = actual in (0, 1) and allocated <= 0
                failed_allocations += int(failed)

                # Post-action diagnostic only. The request was NOT observable
                # by either policy when selecting its action.
                deferred_feasible = actual == 2 and remaining >= request
                feasible_but_deferred += int(deferred_feasible)

            step_rows.append({
                "task": task,
                "policy": policy,
                "training_seed": seed,
                "evaluation_seed": evaluation_seed,
                "step": step + 1,
                "data_mode": mode_description(env),
                "proposed_action": names[proposed],
                "executed_action": names[actual],
                "safety_override": actual != proposed,
                "native_reward": native,
                "evaluated_reward": selected,
                "components_available": components_available,
                "empty_emergency_action": empty_er,
                "normal_patient_served": served_normal,
                "remaining_count_already_zero": zero_remaining,
                "allocation_failed": failed,
                "deferred_full_request_was_feasible": deferred_feasible,
                "request_size": request,
                "allocated_amount": allocated,
                "state_before": json.dumps(before),
                "state_after": json.dumps(after),
            })

            obs = nxt
            if terminated or truncated:
                break

        native_rewards.append(native_total)
        selected_rewards.append(selected_total)
        episode_lengths.append(step + 1)

    label = f"{task}_{policy}_{seed if seed != '' else 'baseline'}"
    write_csv(output / f"{label}_steps.csv", step_rows)

    return {
        "task": task,
        "policy": policy,
        "training_seed": seed,
        "data_mode": mode_description(env),
        "episodes": len(selected_rewards),
        "mean_steps": mean(episode_lengths),
        "native_mean_reward": mean(native_rewards),
        "evaluated_mean_reward": mean(selected_rewards),
        "safety_overrides": overrides,
        "empty_emergency_actions": empty_emergency_actions,
        "normal_patients_served": normal_service,
        "steps_after_remaining_count_zero": zero_remaining_steps,
        "failed_allocations": failed_allocations,
        "feasible_full_requests_deferred": feasible_but_deferred,
        "executed_action_counts": json.dumps(dict(actions)),
    }


def compare_recorded_metrics(folder):
    """Check whether seed results are identical episode by episode."""
    with (folder / "episodes.csv").open(
        newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    comparisons = []
    tasks = sorted({row["task"] for row in rows})

    for task in tasks:
        task_rows = [
            row for row in rows
            if row["task"] == task and row["policy"] == "q_learning"
        ]
        seeds = sorted({row["training_seed"] for row in task_rows})

        if not seeds:
            continue

        reference = {
            row["evaluation_seed"]: row
            for row in task_rows if row["training_seed"] == seeds[0]
        }

        for seed in seeds[1:]:
            candidate = {
                row["evaluation_seed"]: row
                for row in task_rows if row["training_seed"] == seed
            }
            matching_episodes = set(reference) == set(candidate)

            for metric in (
                "reward", "starvation", "jain_fairness", "key_metric"
            ):
                available = all(
                    row[metric] != ""
                    for row in list(reference.values())
                    + list(candidate.values())
                )

                identical = None
                if matching_episodes and available:
                    identical = all(
                        np.isclose(
                            float(reference[episode][metric]),
                            float(candidate[episode][metric]),
                            rtol=0,
                            atol=1e-12,
                        )
                        for episode in reference
                    )

                comparisons.append({
                    "task": task,
                    "reference_seed": seeds[0],
                    "compared_seed": seed,
                    "metric": metric,
                    "matching_episode_sets": matching_episodes,
                    "metric_available": available,
                    "identical_episode_by_episode": identical,
                })

    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    cli = parser.parse_args()

    config = json.loads(
        (cli.results / "configuration.json").read_text(encoding="utf-8")
    )
    args = SimpleNamespace(**config)

    output = cli.results.parent / datetime.now().strftime(
        "audit_%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=False)

    summaries = []

    for task in args.tasks:
        print(f"Auditing {task}: rule baseline", flush=True)
        summaries.append(
            audit_policy(task, "rule", "", None, args, output)
        )

        for seed in args.seeds:
            print(f"Auditing {task}: seed {seed}", flush=True)
            q = np.load(
                cli.results / f"{task}_seed_{seed}.npy",
                allow_pickle=False,
            )
            expected = experiment.q_shape(task, experiment.make_env(task))
            if q.shape != expected:
                raise ValueError(
                    f"{task}: saved shape {q.shape}, expected {expected}"
                )

            summaries.append(
                audit_policy(task, "q_learning", seed, q, args, output)
            )

        write_csv(output / "diagnostic_summary.csv", summaries)

    write_csv(
        output / "seed_metric_identity.csv",
        compare_recorded_metrics(cli.results),
    )

    print(f"\nAudit saved to: {output.resolve()}")


if __name__ == "__main__":
    main()