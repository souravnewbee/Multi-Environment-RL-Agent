from environments.hospital_env import HospitalEnv

def test_task(task_name):
    print("=" * 50)
    print(f"  TASK: {task_name.upper().replace('_', ' ')}")
    print("=" * 50)
    
    env = HospitalEnv(task=task_name)
    state = env.reset()
    
    print(f"\n  Initial State:")
    for key, value in state.items():
        print(f"    {key}: {value}")
    
    print(f"\n  Available Actions: {env.actions}")
    
    print(f"\n  Running all actions:\n")
    for i, action in enumerate(env.actions):
        env.reset()
        next_state, reward, done, info = env.step(i)
        print(f"  Action    : {action}")
        print(f"  Result    : {info['result']}")
        print(f"  Reward    : {reward}")
        print(f"  New State : {next_state}")
        print(f"  {'-' * 40}")

def main():
    print("\n")
    print("*" * 50)
    print("*   UMORDA — HOSPITAL ENVIRONMENT TEST       *")
    print("*" * 50)
    print("\n")
    
    test_task("bed_allocation")
    print("\n")
    test_task("er_queue")
    print("\n")
    test_task("staff_allocation")
    
    print("\n")
    print("*" * 50)
    print("*   ALL TASKS COMPLETED SUCCESSFULLY         *")
    print("*" * 50)
    print("\n")

if __name__ == "__main__":
    main()