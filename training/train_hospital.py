"""
UMORDA — Hospital Training Script
Trains Q-learning agents for all 3 hospital tasks over 20,000 episodes each.
Prints live progress, reward curves, and convergence info.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from environments.hospital_env import HospitalEnv
from agents.hospital_agent import HospitalAgent

# ── Training config ──────────────────────────────────────────────────────────
EPISODES      = 20000
PRINT_EVERY   = 1000    # print summary every N episodes
CURVE_EVERY   = 2000    # print reward bar chart every N episodes

TASK_CONFIGS = [
    {
        "task":      "bed_allocation",
        "q_shape":   (5, 4, 3),      # (bed_bins, patient_bins, actions)
        "n_actions": 3,
        "alpha":     0.1,
        "gamma":     0.97,
        "epsilon_decay": 0.9995,
    },
    {
        "task":      "er_queue",
        "q_shape":   (4, 5, 2),      # (emergency_bins, normal_bins, actions)
        "n_actions": 2,
        "alpha":     0.1,
        "gamma":     0.97,
        "epsilon_decay": 0.9995,
    },
    {
        "task":      "staff_allocation",
        "q_shape":   (4, 4, 3),      # (doctor_bins, load_bins, actions)
        "n_actions": 3,
        "alpha":     0.08,           # slightly lower LR — more stable
        "gamma":     0.97,
        "epsilon_decay": 0.9997,     # slower decay — needs more exploration
    },
]


# ── Mini bar chart for reward curve ─────────────────────────────────────────
def bar(value, max_val=150, width=30):
    if max_val == 0:
        return ""
    filled = int(width * max(0, value) / max_val)
    return "█" * filled


# ── Train a single task ──────────────────────────────────────────────────────
def train_task(config):
    task = config["task"]
    print(f"\n{'='*54}")
    print(f"  TASK : {task.upper().replace('_', ' ')}")
    print(f"  Episodes: {EPISODES:,}  |  α={config['alpha']}  |  γ={config['gamma']}")
    print(f"{'='*54}")

    env   = HospitalEnv(task=task)
    agent = HospitalAgent(
        task          = task,
        q_shape       = config["q_shape"],
        n_actions     = config["n_actions"],
        alpha         = config["alpha"],
        gamma         = config["gamma"],
        epsilon_decay = config["epsilon_decay"],
    )

    window_reward  = 0.0
    window_deltas  = []
    reward_history = []      # avg reward per PRINT_EVERY block

    for episode in range(1, EPISODES + 1):
        obs, _       = env.reset()
        total_reward = 0.0
        ep_deltas    = []
        done         = False

        while not done:
            action               = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done                 = terminated or truncated
            delta                = agent.update(obs, action, reward, next_obs, done)
            ep_deltas.append(delta)
            total_reward        += reward
            obs                  = next_obs

        agent.decay_epsilon()
        mean_delta = float(np.mean(ep_deltas)) if ep_deltas else 0.0
        agent.log_episode(total_reward, mean_delta)

        window_reward += total_reward
        window_deltas.append(mean_delta)

        # ── Print progress ───────────────────────────────────────────────────
        if episode % PRINT_EVERY == 0:
            avg_r     = window_reward / PRINT_EVERY
            avg_delta = float(np.mean(window_deltas))
            converged = "✓ CONVERGED" if agent.is_converged() else ""
            print(
                f"  Ep {episode:6,} | "
                f"Avg Reward: {avg_r:7.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"ΔQ: {avg_delta:.5f}  {converged}"
            )
            reward_history.append(avg_r)
            window_reward = 0.0
            window_deltas = []

        # ── Print reward curve ───────────────────────────────────────────────
        if episode % CURVE_EVERY == 0:
            print(f"\n  ── Reward Curve (every {PRINT_EVERY} eps) ──────────────")
            max_r = max(reward_history) if reward_history else 1
            for i, r in enumerate(reward_history):
                label = f"Ep {(i+1)*PRINT_EVERY:6,}"
                print(f"  {label} | {bar(r, max_r)} {r:.1f}")
            print()

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n  ── Final Summary ──────────────────────────────")
    agent.summary()

    # Save Q-table
    save_path = f"qtables/hospital_{task}.npy"
    agent.save(save_path)
    print(f"  Q-table saved → {save_path}")

    return agent


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n")
    print("*" * 54)
    print("*   UMORDA — HOSPITAL RL TRAINING               *")
    print(f"*   {EPISODES:,} episodes per task                    *")
    print("*" * 54)

    trained_agents = {}
    for config in TASK_CONFIGS:
        agent = train_task(config)
        trained_agents[config["task"]] = agent

    print("\n")
    print("*" * 54)
    print("*   ALL TASKS TRAINED SUCCESSFULLY              *")
    print("*" * 54)
    print("\n  Q-tables saved in qtables/")
    print("  Run view_qtable.py to inspect decisions.\n")


if __name__ == "__main__":
    main()