import numpy as np
import os

def view_bed_allocation(Q):
    print("\n  Best Action for each State:")
    print(f"  {'Free Beds':<15} {'Waiting Patients':<20} {'Best Action'}")
    print(f"  {'-'*50}")
    
    actions = ["Admit", "Reject", "Transfer"]
    bed_labels     = ["0-4", "5-9", "10-14", "15-19", "20+"]
    patient_labels = ["0-9", "10-19", "20-29", "30+"]
    
    for b, bl in enumerate(bed_labels):
        for p, pl in enumerate(patient_labels):
            best = np.argmax(Q[b][p])
            print(f"  Beds: {bl:<12} Patients: {pl:<15} → {actions[best]}")

def view_er_queue(Q):
    print("\n  Best Action for each State:")
    print(f"  {'Emergency Queue':<20} {'Normal Queue':<20} {'Best Action'}")
    print(f"  {'-'*55}")
    
    actions = ["Serve Emergency", "Serve Normal"]
    emergency_labels = ["0-2", "3-5", "6-8", "9+"]
    normal_labels    = ["0-4", "5-9", "10-14", "15-19", "20+"]
    
    for e, el in enumerate(emergency_labels):
        for n, nl in enumerate(normal_labels):
            best = np.argmax(Q[e][n])
            print(f"  Emergency: {el:<12} Normal: {nl:<15} → {actions[best]}")

def view_staff_allocation(Q):
    print("\n  Best Action for each State:")
    print(f"  {'Doctors':<15} {'Patient Load':<20} {'Best Action'}")
    print(f"  {'-'*50}")
    
    actions = ["Assign More Staff", "Keep Current", "Reduce Staff"]
    doctor_labels = ["1-4", "5-9", "10-14", "15+"]
    load_labels   = ["0-14", "15-29", "30-44", "45+"]
    
    for d, dl in enumerate(doctor_labels):
        for l, ll in enumerate(load_labels):
            best = np.argmax(Q[d][l])
            print(f"  Doctors: {dl:<12} Load: {ll:<15} → {actions[best]}")

def main():
    print("\n")
    print("*" * 50)
    print("*   UMORDA — HOSPITAL Q-TABLE VIEWER         *")
    print("*" * 50)

    tasks = [
        ("bed_allocation",   "BED ALLOCATION",   view_bed_allocation),
        ("er_queue",         "ER QUEUE",         view_er_queue),
        ("staff_allocation", "STAFF ALLOCATION", view_staff_allocation),
    ]

    for filename, title, viewer in tasks:
        path = f"qtables/hospital_{filename}.npy"
        
        print(f"\n{'='*50}")
        print(f"  TASK: {title}")
        print(f"{'='*50}")
        
        if not os.path.exists(path):
            print(f"  Q-table not found. Run train_hospital.py first.")
            continue
        
        Q = np.load(path)
        print(f"  Q-table shape: {Q.shape}")
        print(f"  Total states:  {Q.shape[0] * Q.shape[1]}")
        viewer(Q)

    print("\n")
    print("*" * 50)
    print("*   Q-TABLE VIEWING COMPLETE                 *")
    print("*" * 50)
    print("\n")

if __name__ == "__main__":
    main()