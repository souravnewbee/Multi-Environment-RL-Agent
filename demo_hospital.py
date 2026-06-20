import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from environments.hospital_env import HospitalEnv

def discretize(state, task):
    if task == "bed_allocation":
        beds     = min(state["free_beds"], 20) // 5
        patients = min(state["waiting_patients"], 30) // 10
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
        normal = min(state["normal_queue"], 20) // 5
        return (emergency, normal)
    elif task == "staff_allocation":
        doctors = min(state["available_doctors"], 15) // 5
        load    = min(state["patient_load"], 50) // 15
        return (doctors, load)

def load_qtable(task):
    path = f"qtables/hospital_{task}.npy"
    if not os.path.exists(path):
        print(f"\n  Q-table not found. Run train_hospital.py first.\n")
        return None
    return np.load(path)

def get_recommendation(task, state, Q, env):
    s      = discretize(state, task)
    action = np.argmax(Q[s])
    return env.actions[action]

def demo_bed_allocation():
    print("\n  Enter hospital bed situation:")
    free_beds        = int(input("  Free beds available : "))
    waiting_patients = int(input("  Patients waiting    : "))
    
    state = {"free_beds": free_beds, "waiting_patients": waiting_patients}
    Q     = load_qtable("bed_allocation")
    if Q is None: return
    
    env    = HospitalEnv(task="bed_allocation")

    # Hard rule — override agent if 0 beds
    if free_beds == 0:
        action = "Transfer"
        reason = "No beds available, patient must be transferred"
    else:
        action = get_recommendation("bed_allocation", state, Q, env)
        if action == "Admit":
            reason = "Beds available, admit the patient"
        elif action == "Reject":
            reason = "No capacity, rejection necessary"
        elif action == "Transfer":
            reason = "Transfer to manage load efficiently"
    
    print(f"\n  ── Situation ──────────────────────────")
    print(f"  Free Beds        : {free_beds}")
    print(f"  Waiting Patients : {waiting_patients}")
    print(f"  ── Recommendation ─────────────────────")
    print(f"  Action           : {action}")
    print(f"  Reason           : {reason}")

def demo_er_queue():
    print("\n  Enter ER queue situation:")
    emergency = int(input("  Emergency patients waiting : "))
    normal    = int(input("  Normal patients waiting    : "))
    
    state = {"emergency_queue": emergency, "normal_queue": normal}
    Q     = load_qtable("er_queue")
    if Q is None: return
    
    env    = HospitalEnv(task="er_queue")

    # Hard rules — override agent for edge cases
    if emergency == 0 and normal == 0:
        action = "No Action"
        reason = "No patients waiting in either queue"
    elif emergency == 0:
        action = "Serve Normal"
        reason = "No emergency patients, serve normal queue"
    elif normal == 0:
        action = "Serve Emergency"
        reason = "No normal patients, serve emergency queue"
    else:
        action = get_recommendation("er_queue", state, Q, env)
        if action == "Serve Emergency":
            reason = "Emergency cases take priority"
        elif action == "Serve Normal":
            reason = "No emergency cases, serve normal queue"

    print(f"\n  ── Situation ──────────────────────────")
    print(f"  Emergency Queue  : {emergency}")
    print(f"  Normal Queue     : {normal}")
    print(f"  ── Recommendation ─────────────────────")
    print(f"  Action           : {action}")
    print(f"  Reason           : {reason}")

def demo_staff_allocation():
    print("\n  Enter staff situation:")
    doctors = int(input("  Available doctors : "))
    load    = int(input("  Patient load      : "))
    
    state = {"available_doctors": doctors, "patient_load": load}
    Q     = load_qtable("staff_allocation")
    if Q is None: return
    
    env    = HospitalEnv(task="staff_allocation")
    action = get_recommendation("staff_allocation", state, Q, env)
    
    print(f"\n  ── Situation ──────────────────────────")
    print(f"  Available Doctors : {doctors}")
    print(f"  Patient Load      : {load}")
    print(f"  ── Recommendation ─────────────────────")
    print(f"  Action            : {action}")
    
    if action == "Assign More Staff":
        print(f"  Reason            : High patient load needs more doctors")
    elif action == "Keep Current":
        print(f"  Reason            : Load is balanced, maintain staffing")
    elif action == "Reduce Staff":
        print(f"  Reason            : Low load, reduce cost by cutting staff")

def main():
    print("\n")
    print("*" * 50)
    print("*   UMORDA — HOSPITAL DEMO                   *")
    print("*   Enter values, get AI recommendation      *")
    print("*" * 50)

    while True:
        print("\n  Select a task:")
        print("  1. Bed Allocation")
        print("  2. ER Queue Management")
        print("  3. Staff Allocation")
        print("  4. Exit")
        
        choice = input("\n  Enter choice (1/2/3/4): ").strip()
        
        if choice == "1":
            demo_bed_allocation()
        elif choice == "2":
            demo_er_queue()
        elif choice == "3":
            demo_staff_allocation()
        elif choice == "4":
            print("\n  Exiting demo. Goodbye!\n")
            break
        else:
            print("\n  Invalid choice. Please enter 1, 2, 3 or 4.")

if __name__ == "__main__":
    main()