# =============================================================================
# UMORDA — Energy Domain Training Script
# File: training/train_energy.py
# Episodes: 20,000
# Domain: Balcony Solar Panel (Balkonkraftwerk) Optimization
# =============================================================================

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from environments.energy_env import EnergyEnv
from agents.energy_agent import EnergyAgent

EPISODES      = 20_000
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.9995
PRINT_EVERY   = 2_000
QTABLE_DIR    = os.path.join(os.path.dirname(__file__), "..", "qtables")
TASKS         = ["solar_scheduling", "battery_management", "grid_interaction"]


def train_task(task: str):
    print(f"\n{'#'*65}")
    print(f"#  TRAINING TASK: {task.upper():^43} #")
    print(f"{'#'*65}")

    env       = EnergyEnv(task=task)
    cfg       = EnergyEnv.TASK_CONFIG[task]
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    print(f"\n  Task        : {cfg['description']}")
    print(f"  State vars  : {cfg['state_vars']}")
    print(f"  State space : {n_states} states")
    print(f"  Actions     : {cfg['action_meanings']}")
    print(f"  Episodes    : {EPISODES:,}")
    print(f"  Alpha (α)   : {ALPHA}  |  Gamma (γ): {GAMMA}")
    print(f"  Epsilon (ε) : {EPSILON_START} → {EPSILON_MIN} (decay={EPSILON_DECAY})")

    agent = EnergyAgent(
        n_states=n_states, n_actions=n_actions,
        alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    # Show INITIAL Q-Table
    agent.print_qtable(cfg["action_meanings"],
                       title=f"INITIAL Q-TABLE [{task.upper()}]")

    print(f"\n{'─'*65}")
    print(f"  {'Episode':>10} | {'Total Reward':>13} | {'Avg Reward':>10} | {'Epsilon':>9}")
    print(f"  {'─'*10}-+-{'─'*13}-+-{'─'*10}-+-{'─'*9}")

    reward_history = []

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            done = terminated or truncated

        agent.decay_epsilon()
        reward_history.append(total_reward)

        if episode % PRINT_EVERY == 0:
            avg = np.mean(reward_history[-PRINT_EVERY:])
            print(
                f"  {episode:>10,} | {total_reward:>13.2f} | "
                f"{avg:>10.2f} | {agent.epsilon:>9.6f}"
            )

    print(f"\n{'─'*65}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Final Epsilon : {agent.epsilon:.6f}")
    print(f"  Avg Reward    : {np.mean(reward_history):.4f}")
    print(f"  Best Episode  : {max(reward_history):.2f}")
    print(f"  Worst Episode : {min(reward_history):.2f}")
    print(f"{'─'*65}")

    # Show TRAINED Q-Table
    agent.print_qtable(cfg["action_meanings"],
                       title=f"TRAINED Q-TABLE [{task.upper()}]")

    os.makedirs(QTABLE_DIR, exist_ok=True)
    agent.save_qtable(os.path.join(QTABLE_DIR, f"energy_{task}_qtable.npy"))
    return agent, reward_history


if __name__ == "__main__":
    print("=" * 65)
    print("  UMORDA — ENERGY DOMAIN TRAINING")
    print("  Balcony Solar Panel (Balkonkraftwerk) Optimization")
    print("  Q-Learning with Bellman Equation | 20,000 Episodes")
    print("=" * 65)

    results = {}
    for task in TASKS:
        agent, history = train_task(task)
        results[task]  = {"avg": float(np.mean(history)), "eps": agent.epsilon}

    print(f"\n{'='*65}")
    print(f"  ALL ENERGY TASKS TRAINED SUCCESSFULLY!")
    print(f"{'='*65}")
    print(f"  {'Task':>22} | {'Avg Reward':>12} | {'Final Epsilon':>14}")
    print(f"  {'─'*22}-+-{'─'*12}-+-{'─'*14}")
    for task, res in results.items():
        print(f"  {task:>22} | {res['avg']:>12.4f} | {res['eps']:>14.6f}")
    print(f"\n  Q-Tables saved in: qtables/")
    print(f"  Run: python view_qtable_energy.py   to see explained decisions")
    print(f"  Run: python demo_energy.py           to test live")
    print(f"  Run: python visualize_energy.py      for graphs\n")
