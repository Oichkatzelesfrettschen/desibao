import pmwd
import inspect
import sys

try:
    sig = inspect.signature(pmwd.nbody)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Could not get signature: {e}")

try:
    print(f"Docstring: {pmwd.nbody.__doc__}")
except:
    pass
