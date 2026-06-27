# =============================================================================
# UMORDA — Traffic Domain Training Script (FIXED VERSION v2)
# File: training/train_traffic.py
# Episodes: 20,000
# =============================================================================

import sys, os, numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from environments.traffic_env import TrafficEnv
from agents.traffic_agent import TrafficAgent

EPISODES      = 20_000
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.9995
PRINT_EVERY   = 2_000
QTABLE_DIR    = os.path.join(os.path.dirname(__file__), "..", "qtables")
TASKS         = ["intersection", "pedestrian", "parking"]


def train_task(task: str):
    print(f"\n{'#'*65}")
    print(f"#  TRAINING TASK: {task.upper():^43} #")
    print(f"{'#'*65}")

    env       = TrafficEnv(task=task)
    cfg       = TrafficEnv.TASK_CONFIG[task]
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    print(f"\n  Task        : {cfg['description']}")
    print(f"  State vars  : {cfg['state_vars']}")
    print(f"  State space : {n_states} states")
    print(f"  Actions     : {cfg['action_meanings']}")
    print(f"  Episodes    : {EPISODES:,}")
    print(f"  Alpha (α)   : {ALPHA}  |  Gamma (γ): {GAMMA}")
    print(f"  Epsilon (ε) : {EPSILON_START} → {EPSILON_MIN} (decay={EPSILON_DECAY})")
    print(f"\n  KEY FIXES IN THIS VERSION:")
    print(f"  ✓ State includes WAIT TIME (not just count)")
    print(f"  ✓ Hard safety override (pedestrian/wait limits enforced)")
    print(f"  ✓ Phase memory prevents rapid signal flipping")
    print(f"  ✓ Wider state space handles heavy traffic")
    print(f"  ✓ Reward scales with urgency (wait time based)")

    agent = TrafficAgent(
        n_states=n_states, n_actions=n_actions,
        alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
    )

    # Show initial Q-Table
    agent.print_qtable(cfg["action_meanings"],
                       title=f"INITIAL Q-TABLE [{task.upper()}] (optimistic init=1.0)")

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
            print(f"  {episode:>10,} | {total_reward:>13.2f} | {avg:>10.2f} | {agent.epsilon:>9.6f}")

    print(f"\n{'─'*65}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Final Epsilon : {agent.epsilon:.6f}")
    print(f"  Avg Reward    : {np.mean(reward_history):.4f}")
    print(f"  Best Episode  : {max(reward_history):.2f}")
    print(f"{'─'*65}")

    agent.print_qtable(cfg["action_meanings"],
                       title=f"TRAINED Q-TABLE [{task.upper()}]")

    os.makedirs(QTABLE_DIR, exist_ok=True)
    agent.save_qtable(os.path.join(QTABLE_DIR, f"traffic_{task}_qtable.npy"))
    return agent, reward_history


if __name__ == "__main__":
    print("=" * 65)
    print("  UMORDA — TRAFFIC DOMAIN TRAINING (FIXED v2)")
    print("  Fixes: Wait time states, Hard safety, Phase memory,")
    print("         Wider state space, Urgency-based rewards")
    print("=" * 65)

    results = {}
    for task in TASKS:
        agent, history = train_task(task)
        results[task]  = {"avg": float(np.mean(history)), "eps": agent.epsilon}

    print(f"\n{'='*65}")
    print(f"  ALL TASKS TRAINED SUCCESSFULLY!")
    print(f"{'='*65}")
    print(f"  {'Task':>15} | {'Avg Reward':>12} | {'Final Epsilon':>14}")
    print(f"  {'─'*15}-+-{'─'*12}-+-{'─'*14}")
    for task, res in results.items():
        print(f"  {task:>15} | {res['avg']:>12.4f} | {res['eps']:>14.6f}")
    print(f"\n  Q-Tables saved in: qtables/")
    print(f"  Run: python view_qtable_traffic.py  to see explained decisions")
    print(f"  Run: python demo_traffic.py         to test live")
    print(f"  Run: python visualize_traffic.py    for graphs\n")
