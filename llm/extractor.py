"""
UMORDA — llm/extractor.py

README.md documents this file as the module that "Extracts state variables
from natural language". The actual extraction logic lives in
llm_client.extract_state() at the repo root. See llm/router.py for why
this is a thin re-export rather than a second copy of the logic.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import extract_state, TASK_FIELD_SPECS, SCALE_CONTEXT

__all__ = ["extract_state", "TASK_FIELD_SPECS", "SCALE_CONTEXT"] 
