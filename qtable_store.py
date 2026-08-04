"""
UMORDA — Q-Table Database Storage
File: qtable_store.py

Replaces scattered qtables/*.npy files with ONE portable SQLite database
(qtables/qtables.db). Solves the "training gets lost when the project is
transferred" problem: instead of remembering to copy/commit a dozen .npy
files, you only ever have to keep track of one .db file.

Works as a drop-in replacement for np.save / np.load in every agent
(hospital_agent.py, energy_agent.py, traffic_agent.py, finance_agent.py).

Usage
-----
    from qtable_store import save_qtable, load_qtable

    save_qtable("hospital_bed_allocation", Q)          # instead of np.save(...)
    Q = load_qtable("hospital_bed_allocation")         # instead of np.load(...)

No server, no extra dependencies (sqlite3 is in the Python standard library).
If you later need MULTIPLE machines sharing the SAME live Q-table at once
(not just transferring a snapshot), that's a different problem — you'd want
a real client/server DB (Postgres/MySQL) instead of SQLite. Ask if that's
actually what you need and I can wire that up instead.
"""

import sqlite3
import numpy as np
import os
import datetime

DEFAULT_DB_PATH = os.path.join("qtables", "qtables.db")


def _connect(db_path=DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qtables (
            name       TEXT PRIMARY KEY,
            shape      TEXT NOT NULL,
            dtype      TEXT NOT NULL,
            data       BLOB NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    return conn


def save_qtable(name: str, Q: np.ndarray, db_path: str = DEFAULT_DB_PATH):
    """Save a Q-table (numpy array) under `name`, e.g. 'hospital_bed_allocation'."""
    conn = _connect(db_path)
    shape_str = ",".join(str(d) for d in Q.shape)
    conn.execute(
        """INSERT INTO qtables (name, shape, dtype, data, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
               shape=excluded.shape,
               dtype=excluded.dtype,
               data=excluded.data,
               updated_at=excluded.updated_at""",
        (name, shape_str, str(Q.dtype), Q.tobytes(),
         datetime.datetime.now(datetime.timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()
    print(f"  [✓] Q-table '{name}' saved to {db_path}")


def load_qtable(name: str, db_path: str = DEFAULT_DB_PATH) -> np.ndarray:
    """Load a Q-table previously saved under `name`. Raises if not found."""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT shape, dtype, data FROM qtables WHERE name = ?", (name,)
    ).fetchone()
    conn.close()

    if row is None:
        raise FileNotFoundError(
            f"No Q-table named '{name}' in {db_path}. "
            f"Run the matching training script first."
        )

    shape_str, dtype_str, blob = row
    shape = tuple(int(x) for x in shape_str.split(","))
    return np.frombuffer(blob, dtype=dtype_str).reshape(shape).copy()


def list_qtables(db_path: str = DEFAULT_DB_PATH) -> list:
    """List all Q-table names currently stored, with shape and last-updated time."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT name, shape, updated_at FROM qtables ORDER BY name"
    ).fetchall()
    conn.close()
    return [{"name": r[0], "shape": r[1], "updated_at": r[2]} for r in rows]


def delete_qtable(name: str, db_path: str = DEFAULT_DB_PATH):
    conn = _connect(db_path)
    conn.execute("DELETE FROM qtables WHERE name = ?", (name,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Quick self-test
    print("Running qtable_store self-test...")
    test_Q = np.random.uniform(-1, 1, size=(5, 4, 3))
    save_qtable("selftest_task", test_Q)
    loaded = load_qtable("selftest_task")
    assert np.allclose(test_Q, loaded), "Round-trip mismatch!"
    print("Round-trip OK. Stored tables:")
    for entry in list_qtables():
        print(" ", entry)
    delete_qtable("selftest_task")
    print("Self-test cleaned up.")
