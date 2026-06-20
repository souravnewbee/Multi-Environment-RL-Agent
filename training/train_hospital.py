import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
from environments.hospital_env import HospitalEnv

# -- Settings --------------------------------------
EPISODES      = 20000   # full shifts, not single decisions
ALPHA         = 0.1     # learning rate
GAMMA         = 0.95    # discount factor (matters now -- real multi-step env)
EPSILON       = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.9995  # slower decay so 20k episodes keeps exploring longer

# -- Discretize state into bins ---------------------
def discretize(state, task):
    if task == "bed_allocation":
        beds     = min(state["free_beds"], 20) // 5         # 0-3 (4 buckets: 0-4,5-9,10-14,15-19/20+ collapsed by min)
        patients = min(state["waiting_patients"], 30) // 10 # 0-3
        return (beds, patients)

    elif task == "er_queue":
        eq = state["emergency_queue"]
        if eq == 0:
            emergency = 0
        elif eq <= 2:
            emergency = 1
        elif eq <= 5:
            emergency = 2
        elif eq <= 8:
            emergency = 3
        else:
            emergency = 4
        normal = min(state["normal_queue"], 20) // 5         # 0-4
        return (emergency, normal)

    elif task == "staff_allocation":
        doctors = min(state["available_doctors"], 15) // 5  # 0-3
        load    = min(state["patient_load"], 50) // 15      # 0-3
        return (doctors, load)

# -- Train one task ---------------------------------
def train_task(task_name, q_shape, n_actions):
    print(f"\n  Training: {task_name.upper().replace('_', ' ')}")
    print(f"  {'-' * 40}")

    env     = HospitalEnv(task=task_name)
    Q       = np.zeros(q_shape)
    visits  = np.zeros(q_shape, dtype=int)  # how many times each (state, action) was updated
    epsilon = EPSILON

    reward_per_block = []
    total_reward     = 0
    block_size       = max(1, EPISODES // 20)

    for episode in range(1, EPISODES + 1):
        state   = env.reset()
        s       = discretize(state, task_name)
        ep_reward = 0

        done = False
        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, n_actions - 1)  # explore
            else:
                action = int(np.argmax(Q[s]))              # exploit

            next_state, reward, done, info = env.step(action)
            ns = discretize(next_state, task_name)

            # Q-table update (real multi-step bootstrap now that the
            # next state actually follows from this action within the shift)
            best_next = 0.0 if done else np.max(Q[ns])
            Q[s][action] = Q[s][action] + ALPHA * (
                reward + GAMMA * best_next - Q[s][action]
            )
            visits[s][action] += 1

            ep_reward += reward
            s = ns

        total_reward += ep_reward
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % block_size == 0:
            avg = total_reward / block_size
            reward_per_block.append(avg)
            print(f"  Episode {episode:6d} | Avg Shift Reward: {avg:8.2f} | Epsilon: {epsilon:.3f}")
            total_reward = 0

    print(f"\n  Reward progression:")
    for i, r in enumerate(reward_per_block):
        bar_len = int(max(0, r) / 5)
        bar = "#" * bar_len
        print(f"  Block {(i+1)*block_size:6d} | {bar} {r:.2f}")

    unvisited = int(np.sum(np.sum(visits, axis=-1) == 0))
    total_states = int(np.prod(q_shape[:-1]))
    print(f"\n  State coverage: {total_states - unvisited}/{total_states} states visited at least once")
    if unvisited > 0:
        print(f"  WARNING: {unvisited} states never visited -- their Q-values are still zero-init guesses")

    # Save Q-table and visit counts
    os.makedirs("qtables", exist_ok=True)
    np.save(f"qtables/hospital_{task_name}.npy", Q)
    np.save(f"qtables/hospital_{task_name}_visits.npy", visits)
    print(f"\n  Q-table saved -> qtables/hospital_{task_name}.npy")
    print(f"  Visit counts saved -> qtables/hospital_{task_name}_visits.npy")

    return Q, visits

# -- Main --------------------------------------------
def main():
    print("\n")
    print("*" * 50)
    print("*   UMORDA -- HOSPITAL TRAINING                *")
    print("*" * 50)

    train_task("bed_allocation",   q_shape=(5, 4, 3), n_actions=3)
    train_task("er_queue",         q_shape=(5, 5, 2), n_actions=2)
    train_task("staff_allocation", q_shape=(4, 4, 3), n_actions=3)

    print("\n")
    print("*" * 50)
    print("*   ALL TASKS TRAINED SUCCESSFULLY            *")
    print("*" * 50)
    print("\n")

if __name__ == "__main__":
    main()
