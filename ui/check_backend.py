"""
UMORDA — Backend Sanity Check
File: ui/check_backend.py

Quick standalone script to verify ui/app.py is actually working, without
needing a browser. Hits each endpoint in turn and reports pass/fail with
enough detail to debug (e.g. missing GROQ_API_KEY, missing Q-tables).

Usage:
    1. In one terminal: python ui/app.py
    2. In another:      python ui/check_backend.py
"""

import sys
import json
import requests

BASE = "http://localhost:8000"


def ok(msg):
    print(f"  [OK] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def main():
    print("\n" + "=" * 55)
    print("  UMORDA — Backend Check")
    print("=" * 55)

    # ── 1. Is the server even up? ─────────────────────────────────────────
    try:
        r = requests.get(f"{BASE}/api/health", timeout=5)
    except requests.exceptions.ConnectionError:
        fail(f"Could not connect to {BASE}")
        print("       Is `python ui/app.py` running in another terminal?")
        sys.exit(1)

    if r.status_code == 200 and r.json().get("status") == "ok":
        ok("GET /api/health")
    else:
        fail(f"GET /api/health returned {r.status_code}: {r.text}")

    # ── 2. Domains ─────────────────────────────────────────────────────────
    r = requests.get(f"{BASE}/api/domains", timeout=5)
    if r.status_code == 200 and isinstance(r.json(), dict) and len(r.json()) > 0:
        ok(f"GET /api/domains -> {list(r.json().keys())}")
    else:
        fail(f"GET /api/domains returned {r.status_code}: {r.text}")

    # ── 3. History (should work even if empty) ────────────────────────────
    r = requests.get(f"{BASE}/api/history", timeout=5)
    if r.status_code == 200 and isinstance(r.json(), list):
        ok(f"GET /api/history -> {len(r.json())} entries")
    else:
        fail(f"GET /api/history returned {r.status_code}: {r.text}")

    # ── 4. The real test: a full pipeline query ────────────────────────────
    # This is the one that actually needs GROQ_API_KEY (or LLM_BACKEND=ollama)
    # and a trained Q-table for the routed task.
    test_query = "The ER has 8 emergency patients and 3 normal patients waiting"
    print(f"\n  Sending test query: \"{test_query}\"")
    try:
        r = requests.post(
            f"{BASE}/api/query",
            json={"query": test_query, "domain": None},
            timeout=30,
        )
    except requests.exceptions.Timeout:
        fail("POST /api/query timed out (30s) — LLM backend may be unreachable")
        sys.exit(1)

    if r.status_code == 200:
        data = r.json()
        ok("POST /api/query")
        print(f"       Domain      : {data['route']['domain_label']}")
        print(f"       Task        : {data['extract']['task']}")
        print(f"       Action      : {data['decide']['chosen_action']}")
        print(f"       Confidence  : {round(data['decide']['confidence'] * 100)}%")
        print(f"       Q-values    : {data['decide']['q_values']}")
        print(f"       Explanation : {data['explain']}")
    elif r.status_code == 422:
        fail(f"POST /api/query -> 422: {r.json().get('detail')}")
        print("       (Router couldn't match the query, or extractor needs clarification)")
    elif r.status_code == 500:
        fail(f"POST /api/query -> 500: {r.json().get('detail')}")
        print("       Common causes:")
        print("         - GROQ_API_KEY not set (route_message/extract_state/explain_decision fail)")
        print("         - Q-table not trained yet for the routed task (run training/train_*.py)")
    else:
        fail(f"POST /api/query returned {r.status_code}: {r.text}")

    print("\n" + "=" * 55)
    print("  Check complete.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
