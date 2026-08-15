"""
UMORDA — FastAPI Install Check
File: ui/check_fastapi.py

Run this BEFORE ui/app.py to confirm fastapi/uvicorn/pydantic are actually
importable in whatever Python environment you're using. Catches the classic
"pip installed it but into a different Python/venv" problem early.

Usage:
    python ui/check_fastapi.py
"""

import sys

print(f"Python executable: {sys.executable}")
print(f"Python version   : {sys.version.split()[0]}\n")

all_ok = True

def check(name, import_fn):
    global all_ok
    try:
        mod = import_fn()
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK]   {name:<12} {version}")
    except ImportError as e:
        all_ok = False
        print(f"  [FAIL] {name:<12} NOT INSTALLED -> {e}")

check("fastapi",   lambda: __import__("fastapi"))
check("uvicorn",   lambda: __import__("uvicorn"))
check("pydantic",  lambda: __import__("pydantic"))
check("starlette", lambda: __import__("starlette"))

print()
if all_ok:
    print("  All good — you can run: python ui/app.py")
else:
    print("  Missing packages. Install with:")
    print(f"    {sys.executable} -m pip install fastapi \"uvicorn[standard]\" pydantic")
    print("  (using -m pip via sys.executable above avoids installing into the wrong Python)")
