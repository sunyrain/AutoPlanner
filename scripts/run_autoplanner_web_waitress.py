"""Run the AutoPlanner web UI with Waitress for demo use."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The web UI is interactive and normally serves one user. Inheriting the
# benchmark environment's large BLAS thread pools makes an idle CUDA/torch
# process look busy and can slow down route-search requests.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.web.app import create_app


if __name__ == "__main__":
    host = os.environ.get("AUTOPLANNER_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("AUTOPLANNER_WEB_PORT", "7860"))
    serve(create_app(), host=host, port=port, threads=2, channel_timeout=30)
