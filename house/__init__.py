"""L-shaped house and environment construction helpers."""

from pathlib import Path
import sys

_local_uaibot = Path(__file__).resolve().parents[1] / "UAIbotPy"
if str(_local_uaibot) not in sys.path:
    sys.path.insert(0, str(_local_uaibot))
