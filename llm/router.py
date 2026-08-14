"""
UMORDA — llm/router.py

README.md documents this file as the module that "Classifies user prompt
into correct domain + task". The actual routing logic lives in
llm_client.route_message() at the repo root (it's used directly by
main.py, hospital_assistant.py, energy_assistant.py, traffic_assistant.py,
and interactive_demo.py already).

This file was previously empty, which meant `from llm.router import
route_message` -- the import the README's architecture diagram implies --
did not work. It's a thin re-export now: single source of truth stays in
llm_client.py (avoids two copies of the same prompt drifting apart), but
`llm.router` is now a real, working import path.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import route_message, TASK_FIELD_SPECS, TASK_DESCRIPTIONS

__all__ = ["route_message", "TASK_FIELD_SPECS", "TASK_DESCRIPTIONS"]
