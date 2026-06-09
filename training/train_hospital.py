import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
from environments.hospital_env import HospitalEnv

# ── Settings ────────────────────────────────────
EPISODES    = 500
ALPHA       = 0.1   # learning rate
GAMMA       = 0.9   # discount factor
EPSILON     = 1.0   # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# ── Discretize state into bins ───────────────────
def discretize(state, task):
    if task == "bed_allocation":
        beds     = min(state["free_beds"], 20) // 5         # 0-4
        patients = min(state["waiting_patients"], 30) // 10 # 0-3
        return (beds, patients)

    elif task == "er_queue":
        emergency = min(state["emergency_queue"], 10) // 3  # 0-3
        normal    = min(state["normal_queue"], 20) // 5     # 0-4
        return (emergency, normal)

    elif task == "staff_allocation":
        doctors  = min(state["available_doctors"], 15) // 5 # 0-3
        load     = min(state["patient_load"], 50) // 15     # 0-3
        return (doctors, load)

# ── Train one task ───────────────────────────────
def train_task(task_name, q_shape, n_actions):
    print(f"\n  Training: {task_name.upper().replace('_', ' ')}")
    print(f"  {'-' * 40}")

    env     = HospitalEnv(task=task_name)
    Q       = np.zeros(q_shape)
    epsilon = EPSILON

    reward_per_100 = []
    total_reward   = 0

    for episode in range(1, EPISODES + 1):
        state  = env.reset()
        s      = discretize(state, task_name)

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, n_actions - 1)  # explore
        else:
            action = np.argmax(Q[s])                   # exploit

        next_state, reward, done, info = env.step(action)
        ns = discretize(next_state, task_name)

        # Q-table update
        Q[s][action] = Q[s][action] + ALPHA * (
            reward + GAMMA * np.max(Q[ns]) - Q[s][action]
        )

        total_reward += reward
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Print progress every 100 episodes
        if episode % 100 == 0:
            avg = total_reward / 100
            reward_per_100.append(avg)
            print(f"  Episode {episode:4d} | Avg Reward: {avg:6.2f} | Epsilon: {epsilon:.3f}")
            total_reward = 0

    print(f"\n  Reward progression (every 100 episodes):")
    for i, r in enumerate(reward_per_100):
        bar = "█" * int(max(0, r))
        print(f"  Ep {(i+1)*100:4d} | {bar} {r:.2f}")

    # Save Q-table
    os.makedirs("qtables", exist_ok=True)
    np.save(f"qtables/hospital_{task_name}.npy", Q)
    print(f"\n  Q-table saved → qtables/hospital_{task_name}.npy")

    return Q

# ── Main ─────────────────────────────────────────
def main():
    print("\n")
    print("*" * 50)
    print("*   UMORDA — HOSPITAL TRAINING                *")
    print("*" * 50)

    train_task("bed_allocation",   q_shape=(5, 4, 3), n_actions=3)
    train_task("er_queue",         q_shape=(4, 5, 2), n_actions=2)
    train_task("staff_allocation", q_shape=(4, 4, 3), n_actions=3)

    print("\n")
    print("*" * 50)
    print("*   ALL TASKS TRAINED SUCCESSFULLY            *")
    print("*" * 50)
    print("\n")

if __name__ == "__main__":
    main()
