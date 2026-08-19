"""
Run this next to your real llm_client.py, with GROQ_API_KEY (or LLM_BACKEND=ollama)
set, to verify the intensifier-language fix actually behaves as expected against
the real model. This calls extract_state() directly -- no mocking.

    python test_extract_real.py
"""
import json
from llm_client import extract_state

TESTS = [
    ("battery_management", "battery is nearly empty, solar is weak, grid is expensive",
     {"battery_level": 5, "solar_output": 5, "grid_price": 1, "home_consumption": 3},
     False, "intensified 'nearly empty' should resolve directly, no clarification"),

    ("soil_preparation", "soil is really acidic and I only have 5 days left before planting",
     {"soil_ph": 6, "organic_matter": 30, "drainage_quality": 30, "days_remaining": 25},
     False, "intensified 'really acidic' should resolve directly, no clarification"),

    ("battery_management", "battery is low",
     {"battery_level": 5, "solar_output": 5, "grid_price": 1, "home_consumption": 3},
     True, "CONTROL: bare 'low' (no intensifier) should still ask for clarification"),

    ("soil_preparation", "soil is acidic",
     {"soil_ph": 6, "organic_matter": 30, "drainage_quality": 30, "days_remaining": 25},
     True, "CONTROL: bare 'acidic' (no intensifier) should still ask for clarification"),

    ("battery_management", "battery is really high, plenty of solar, grid price is cheap",
     {"battery_level": 5, "solar_output": 0, "grid_price": 1, "home_consumption": 3},
     None, "battery should resolve via intensifier; solar_output is NOT yet covered "
           "by this fix so it may still ask -- known gap, see chat notes"),
]

for task, msg, known_state, expect_clarify, note in TESTS:
    r = extract_state(task, msg, known_state)
    got = r["needs_clarification"]
    status = "PASS" if (expect_clarify is None or got == expect_clarify) else "FAIL"
    print(f"\n[{status}] {task}: \"{msg}\"")
    print(f"  note: {note}")
    print(f"  needs_clarification: {got}  (expected: {expect_clarify})")
    print(f"  state: {r['state']}")
    if r.get("clarification_question"):
        print(f"  clarification_question: {r['clarification_question']}")