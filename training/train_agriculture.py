"""
UMORDA — Agriculture Training Script (Gymnasium-compatible)
Trains Q-learning agents for all 3 agriculture tasks over 20,000 season-episodes
each. soil_preparation runs until planted/window closes (<=25 steps); irrigation
and pest_control run a full 40-step season (AgricultureEnv.SHIFT_LENGTH).

CHANGED: Q-tables are now saved via agent.save() → qtable_store.py (SQLite),
landing in qtables/qtables.db instead of a standalone .npy file.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from environments.agriculture_env import AgricultureEnv
from agents.agriculture_agent     import AgricultureAgent, Q_SHAPES

EPISODES    = 20000
PRINT_EVERY = 1000
CURVE_EVERY = 4000

TASK_CONFIGS = [
    {"task": "soil_preparation", "n_actions": 4, "alpha": 0.1,  "gamma": 0.95, "epsilon_decay": 0.9995},
    {"task": "irrigation",       "n_actions": 3, "alpha": 0.1,  "gamma": 0.95, "epsilon_decay": 0.9995},
    {"task": "pest_control",     "n_actions": 3, "alpha": 0.08, "gamma": 0.95, "epsilon_decay": 0.9997},
]


def bar(value, max_val=600, width=30):
    if max_val == 0:
        return ""
    filled = int(width * max(0, value) / max_val)
    return "█" * filled


def train_task(config):
    task = config["task"]
    q_shape = Q_SHAPES[task]
    print(f"\n{'='*54}")
    print(f"  TASK : {task.upper().replace('_', ' ')}")
    print(f"  Episodes: {EPISODES:,} seasons  |  α={config['alpha']}  |  γ={config['gamma']}")
    print(f"{'='*54}")

    env   = AgricultureEnv(task=task)
    agent = AgricultureAgent(
        task=task, n_actions=config["n_actions"],
        alpha=config["alpha"], gamma=config["gamma"], epsilon_decay=config["epsilon_decay"],
    )

    window_reward  = 0.0
    window_deltas  = []
    reward_history = []

    for episode in range(1, EPISODES + 1):
        obs, _       = env.reset()
        total_reward = 0.0
        ep_deltas    = []
        done         = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done   = terminated or truncated
            delta  = agent.update(obs, action, reward, next_obs, done)
            ep_deltas.append(delta)
            total_reward += reward
            obs = next_obs

        agent.decay_epsilon()
        mean_delta = float(np.mean(ep_deltas)) if ep_deltas else 0.0
        agent.log_episode(total_reward, mean_delta)

        window_reward += total_reward
        window_deltas.append(mean_delta)

        if episode % PRINT_EVERY == 0:
            avg_r     = window_reward / PRINT_EVERY
            avg_delta = float(np.mean(window_deltas))
            converged = "✓ CONVERGED" if agent.is_converged() else ""
            print(f"  Ep {episode:6,} | Avg Reward: {avg_r:8.2f} | ε: {agent.epsilon:.4f} | ΔQ: {avg_delta:.5f}  {converged}")
            reward_history.append(avg_r)
            window_reward = 0.0
            window_deltas = []

        if episode % CURVE_EVERY == 0:
            print(f"\n  ── Reward Curve (every {PRINT_EVERY} eps) ──────────────")
            max_r = max(reward_history) if reward_history else 1
            for i, r in enumerate(reward_history):
                print(f"  Ep {(i+1)*PRINT_EVERY:6,} | {bar(r, max_r)} {r:.1f}")
            print()

    print(f"\n  ── Final Summary ──────────────────────────────")
    agent.summary()

    # CHANGED: was np.save(f"qtables/agriculture_{task}.npy", agent.Q) via agent.save(path)
    # now goes through qtable_store.py -> qtables/qtables.db
    agent.save()
    print(f"  Q-table saved → qtables.db (as 'agriculture_{task}')")

    os.makedirs("logs", exist_ok=True)
    np.savez(f"logs/history_{task}.npz",
        episode_rewards=np.array(agent.episode_rewards),
        episode_epsilons=np.array(agent.episode_epsilons),
        convergence_delta=np.array(agent.convergence_delta))
    print(f"  Training history saved → logs/history_{task}.npz")

    return agent


def main():
    print("\n")
    print("*" * 54)
    print("*   UMORDA — AGRICULTURE RL TRAINING (Gymnasium)*")
    print(f"*   {EPISODES:,} season-episodes per task              *")
    print("*" * 54)

    for config in TASK_CONFIGS:
        train_task(config)

    print("\n")
    print("*" * 54)
    print("*   ALL TASKS TRAINED SUCCESSFULLY              *")
    print("*" * 54)
    print("\n  Q-tables saved in qtables/qtables.db\n")


if __name__ == "__main__":
    main()