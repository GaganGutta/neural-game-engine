import sys
from pathlib import Path

# Let the tests run from a clean checkout without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
