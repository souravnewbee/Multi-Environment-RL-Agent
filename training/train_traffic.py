# =============================================================================
# UMORDA — Traffic Domain Training Script
# File: training/train_traffic.py
# Episodes: 20,000
# =============================================================================

import sys
import os
import numpy as np

# Allow imports from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from environments.traffic_env import TrafficEnv
from agents.traffic_agent import TrafficAgent

# =============================================================================
# CONFIGURATION
# =============================================================================
EPISODES      = 20_000
ALPHA         = 0.1       # Learning rate
GAMMA         = 0.95      # Discount factor
EPSILON_START = 1.0       # Start fully exploring
EPSILON_MIN   = 0.01      # Never stop exploring completely
EPSILON_DECAY = 0.9995    # Decay per episode
PRINT_EVERY   = 1_000     # Print progress every N episodes
QTABLE_DIR    = os.path.join(os.path.dirname(__file__), "..", "qtables")

TASKS = ["intersection", "pedestrian", "parking"]


# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_task(task: str):
    print(f"\n{'#'*65}")
    print(f"#  TRAINING TASK: {task.upper():^43} #")
    print(f"{'#'*65}")

    # --- Environment & Agent ---
    env   = TrafficEnv(task=task)
    cfg   = TrafficEnv.TASK_CONFIG[task]
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    agent = TrafficAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    print(f"\n  Task        : {cfg['description']}")
    print(f"  States      : {n_states}")
    print(f"  Actions     : {n_actions}  → {cfg['action_meanings']}")
    print(f"  Episodes    : {EPISODES:,}")
    print(f"  Alpha (α)   : {ALPHA}")
    print(f"  Gamma (γ)   : {GAMMA}")
    print(f"  Epsilon (ε) : {EPSILON_START} → {EPSILON_MIN}  (decay={EPSILON_DECAY})")

    # --- Show INITIAL Q-Table ---
    agent.print_qtable(cfg["action_meanings"], title=f"INITIAL Q-TABLE  [{task.upper()}]")

    # --- Training Loop ---
    print(f"\n{'─'*65}")
    print(f"  TRAINING IN PROGRESS — {EPISODES:,} EPISODES")
    print(f"{'─'*65}")
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

        # Print progress
        if episode % PRINT_EVERY == 0:
            avg_reward = np.mean(reward_history[-PRINT_EVERY:])
            print(
                f"  {episode:>10,} | {total_reward:>13.2f} | "
                f"{avg_reward:>10.2f} | {agent.epsilon:>9.6f}"
            )

    # --- Training Summary ---
    print(f"\n{'─'*65}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Total Episodes  : {EPISODES:,}")
    print(f"  Final Epsilon   : {agent.epsilon:.6f}")
    print(f"  Avg Reward      : {np.mean(reward_history):.4f}")
    print(f"  Best Episode R  : {max(reward_history):.2f}")
    print(f"  Worst Episode R : {min(reward_history):.2f}")
    print(f"{'─'*65}")

    # --- Show UPDATED Q-Table ---
    agent.print_qtable(cfg["action_meanings"], title=f"TRAINED Q-TABLE  [{task.upper()}]")

    # --- Save Q-Table ---
    os.makedirs(QTABLE_DIR, exist_ok=True)
    save_path = os.path.join(QTABLE_DIR, f"traffic_{task}_qtable.npy")
    agent.save_qtable(save_path)

    return agent, reward_history


# =============================================================================
# MAIN — Train all 3 tasks
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  UMORDA — TRAFFIC DOMAIN TRAINING")
    print("  Multi-Objective Reinforcement Learning Agent")
    print("  Q-Learning with Bellman Equation")
    print("=" * 65)

    results = {}
    for task in TASKS:
        agent, history = train_task(task)
        results[task] = {
            "agent": agent,
            "history": history,
            "final_epsilon": agent.epsilon,
            "avg_reward": float(np.mean(history)),
        }

    # --- Final Summary across all tasks ---
    print(f"\n{'='*65}")
    print(f"  ALL TASKS TRAINED SUCCESSFULLY!")
    print(f"{'='*65}")
    print(f"  {'Task':>15} | {'Avg Reward':>12} | {'Final Epsilon':>14}")
    print(f"  {'─'*15}-+-{'─'*12}-+-{'─'*14}")
    for task, res in results.items():
        print(
            f"  {task:>15} | {res['avg_reward']:>12.4f} | {res['final_epsilon']:>14.6f}"
        )
    print(f"{'='*65}")
    print(f"\n  Q-Tables saved in: qtables/")
    print(f"  Run demo_traffic.py to test your agent!\n")
