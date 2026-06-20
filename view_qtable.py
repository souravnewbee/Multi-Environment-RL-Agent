import numpy as np
import os

def _load_visits(path_qtable, shape):
    visits_path = path_qtable.replace(".npy", "_visits.npy")
    if os.path.exists(visits_path):
        return np.load(visits_path)
    return np.zeros(shape, dtype=int)

def view_bed_allocation(Q, visits):
    print("\n  Best Action for each State:")
    print(f"  {'Free Beds':<15} {'Waiting Patients':<20} {'Best Action':<18} {'Visits'}")
    print(f"  {'-'*65}")

    actions = ["Admit", "Reject", "Transfer"]
    bed_labels     = ["0-4", "5-9", "10-14", "15-19", "20+"]
    patient_labels = ["0-9", "10-19", "20-29", "30+"]

    for b, bl in enumerate(bed_labels):
        for p, pl in enumerate(patient_labels):
            best = np.argmax(Q[b][p])
            v = int(np.sum(visits[b][p]))
            flag = "  <-- UNVISITED" if v == 0 else ""
            print(f"  Beds: {bl:<12} Patients: {pl:<15} -> {actions[best]:<15} {v:<8}{flag}")

def view_er_queue(Q, visits):
    print("\n  Best Action for each State:")
    print(f"  {'Emergency Queue':<20} {'Normal Queue':<20} {'Best Action':<18} {'Visits'}")
    print(f"  {'-'*70}")

    actions = ["Serve Emergency", "Serve Normal"]
    emergency_labels = ["0", "1-2", "3-5", "6-8", "9+"]
    normal_labels    = ["0-4", "5-9", "10-14", "15-19", "20+"]

    for e, el in enumerate(emergency_labels):
        for n, nl in enumerate(normal_labels):
            best = np.argmax(Q[e][n])
            v = int(np.sum(visits[e][n]))
            flag = "  <-- UNVISITED" if v == 0 else ""
            print(f"  Emergency: {el:<12} Normal: {nl:<15} -> {actions[best]:<15} {v:<8}{flag}")

def view_staff_allocation(Q, visits):
    print("\n  Best Action for each State:")
    print(f"  {'Doctors':<15} {'Patient Load':<20} {'Best Action':<22} {'Visits'}")
    print(f"  {'-'*70}")

    actions = ["Assign More Staff", "Keep Current", "Reduce Staff"]
    doctor_labels = ["1-4", "5-9", "10-14", "15+"]
    load_labels   = ["0-14", "15-29", "30-44", "45+"]

    for d, dl in enumerate(doctor_labels):
        for l, ll in enumerate(load_labels):
            best = np.argmax(Q[d][l])
            v = int(np.sum(visits[d][l]))
            flag = "  <-- UNVISITED" if v == 0 else ""
            print(f"  Doctors: {dl:<12} Load: {ll:<15} -> {actions[best]:<18} {v:<8}{flag}")

def main():
    print("\n")
    print("*" * 50)
    print("*   UMORDA -- HOSPITAL Q-TABLE VIEWER         *")
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
        visits = _load_visits(path, Q.shape)
        total_visits = int(np.sum(visits))
        unvisited_states = int(np.sum(np.sum(visits, axis=-1) == 0))
        total_states = Q.shape[0] * Q.shape[1]

        print(f"  Q-table shape: {Q.shape}")
        print(f"  Total states:  {total_states}")
        print(f"  Total (state, action) updates during training: {total_visits}")
        print(f"  States never visited: {unvisited_states}/{total_states}")
        viewer(Q, visits)

    print("\n")
    print("*" * 50)
    print("*   Q-TABLE VIEWING COMPLETE                 *")
    print("*" * 50)
    print("\n")

if __name__ == "__main__":
    main()
