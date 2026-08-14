"""
UMORDA — llm/explainer.py

README.md documents this file as the module that "Converts RL action to
human-readable response". The actual explanation logic lives in
llm_client.explain_decision() / explain_ungrounded() at the repo root.
See llm/router.py for why this is a thin re-export rather than a second
copy of the logic.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import explain_decision, explain_ungrounded

__all__ = ["explain_decision", "explain_ungrounded"]