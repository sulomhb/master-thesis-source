"""
Re-capture scores from completed runs whose live capture missed them
(e.g. the 2026-05-07 smoke test where Langfuse propagation lagged behind
the per-rep fetch).

Reads `windows.json` for a list of `{variant, repetition, started_at, finished_at}`
records and queries Langfuse for scores created in each window.
Appends to results/raw_scores.csv (same schema as runner.py).

Usage:
    python rq3/recapture.py --windows rq3/windows.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Reuse runner.py utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import (  # type: ignore[import-not-found]
    load_env_vars,
    fetch_scores_window,
    append_scores,
    ensure_results_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [recapture] %(levelname)s %(message)s")
log = logging.getLogger("rq3.recapture")


def main() -> int:
    p = argparse.ArgumentParser(description="Re-capture scores from completed runs")
    p.add_argument("--windows", type=Path, default=Path(__file__).resolve().parent / "windows.json",
                   help="JSON file: list of {variant, repetition, started_at, finished_at}")
    args = p.parse_args()

    if not args.windows.exists():
        log.error(f"windows file not found: {args.windows}")
        return 2
    windows = json.loads(args.windows.read_text(encoding="utf-8"))
    env = load_env_vars()
    ensure_results_dir()

    total_captured = 0
    for w in windows:
        started = datetime.fromisoformat(w["started_at"])
        finished = datetime.fromisoformat(w["finished_at"])
        log.info(f"Variant {w['variant']} rep {w['repetition']:02d}: window {started.isoformat()} → {finished.isoformat()}")
        scores = fetch_scores_window(env, started, finished)
        n = append_scores(w["variant"], int(w["repetition"]), scores, started, finished)
        log.info(f"  captured {n} scores")
        total_captured += n
    log.info(f"Total captured: {total_captured}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
