"""
RQ3 runner - orchestrates Phase 2.

For each (variant, repetition) pair, runs the dataset evaluator container
against the fixed Phase 1 dataset run, then captures the new scores from
Langfuse and appends them to results/raw_scores.csv.

Usage:
    python rq3/runner.py                          # full Phase 2 (4 variants × 10 reps)
    python rq3/runner.py --dry-run                # print docker commands, don't execute
    python rq3/runner.py --variants B             # one variant only
    python rq3/runner.py --variants B M --reps 3  # quick smoke (subset)

Env requirements:
- LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in repo-root .env
- Docker daemon running, hneti-dataset-eval:rq3 image present
- Azure OpenAI / Anthropic creds in repo-root .env

Run from repo root:
    python rq3/runner.py
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
RQ3_DIR = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / ".env"
CONFIG_FILE = RQ3_DIR / "config.json"
RESULTS_DIR = RQ3_DIR / "results"
RAW_CSV = RESULTS_DIR / "raw_scores.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [runner] %(levelname)s %(message)s",
)
log = logging.getLogger("rq3.runner")


def load_env_vars() -> dict[str, str]:
    """Load .env into a dict (no shell expansion). Used for Langfuse REST auth."""
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        log.error(f".env not found at {ENV_FILE}")
        sys.exit(2)
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        log.error(f"config.json not found at {CONFIG_FILE}")
        sys.exit(2)
    cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if not cfg.get("dataset_run_name"):
        log.error("config.json: dataset_run_name is null. Trigger Phase 1 first and fill in the run name.")
        sys.exit(2)
    return cfg


def langfuse_auth_header(env: dict[str, str]) -> dict[str, str]:
    pk = env.get("LANGFUSE_PUBLIC_KEY", "")
    sk = env.get("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk:
        log.error("Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env")
        sys.exit(2)
    token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def fetch_scores_window(env: dict[str, str], from_ts: datetime, to_ts: datetime) -> list[dict]:
    """Fetch all scores created within [from_ts, to_ts]. Paginates by 100s."""
    host = env.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    headers = langfuse_auth_header(env)
    headers["Accept"] = "application/json"
    all_scores: list[dict] = []
    page = 1
    while True:
        params = {
            "fromTimestamp": from_ts.isoformat().replace("+00:00", "Z"),
            "toTimestamp": to_ts.isoformat().replace("+00:00", "Z"),
            "limit": 100,
            "page": page,
        }
        url = f"{host}/api/public/scores?{urlencode(params)}"
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as e:
            log.error(f"Langfuse score fetch failed (page {page}): {e}")
            break
        rows = payload.get("data", [])
        all_scores.extend(rows)
        meta = payload.get("meta", {})
        total_pages = meta.get("totalPages", 1)
        if page >= total_pages or not rows:
            break
        page += 1
    return all_scores


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def docker_run(
    image: str,
    env_file: Path,
    extra_env: dict[str, str],
    args: list[str],
    dry_run: bool,
) -> int:
    cmd = ["docker", "run", "--rm", "--env-file", str(env_file)]
    for k, v in extra_env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [image] + args
    log.info("docker cmd: " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        with RAW_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "variant", "repetition", "trace_id", "observation_id",
                "metric_name", "score_value", "score_data_type", "score_comment",
                "run_started_at", "run_finished_at", "score_created_at",
            ])


def append_scores(variant_code: str, rep: int, scores: list[dict], started: datetime, finished: datetime) -> int:
    n = 0
    with RAW_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in scores:
            w.writerow([
                variant_code,
                rep,
                s.get("traceId") or "",
                s.get("observationId") or "",
                s.get("name") or "",
                s.get("value"),
                s.get("dataType") or "",
                (s.get("comment") or "")[:1000],
                started.isoformat(),
                finished.isoformat(),
                s.get("createdAt") or "",
            ])
            n += 1
    return n


def run_one_variant_rep(
    variant: dict[str, Any],
    rep: int,
    cfg: dict[str, Any],
    env_vars: dict[str, str],
    dry_run: bool,
) -> tuple[int, int]:
    """Returns (docker_exit_code, scores_captured)."""
    image = cfg["phase2"]["image_tag"]
    item_limit = cfg["phase2"]["item_limit"]
    if cfg.get("_item_limit_override") is not None:
        item_limit = cfg["_item_limit_override"]
    dataset_name = cfg["dataset_name"]
    run_name = cfg["dataset_run_name"]

    extra_env = dict(variant.get("env_overrides") or {})
    # Make Langfuse env tag explicit so we can find the scores later.
    extra_env.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "test")

    args = [
        "--dataset-name", dataset_name,
        "--run-name", run_name,
        "--item-limit", str(item_limit),
        "--add-scores",
        "--experiment-version", f"rq3_{variant['code']}_rep_{rep:02d}",
    ]

    started = now_utc()
    log.info(f"=== Variant {variant['code']} rep {rep:02d} START at {started.isoformat()} ===")
    code = docker_run(image, ENV_FILE, extra_env, args, dry_run)
    finished = now_utc()
    log.info(f"=== Variant {variant['code']} rep {rep:02d} END   at {finished.isoformat()} (exit={code}, dur={(finished-started).total_seconds():.0f}s) ===")

    if dry_run:
        return code, 0

    # Capture scores created during this container's lifetime.
    # Langfuse SDK batches score writes (flush_at=32) and there's a propagation lag
    # of several seconds between container shutdown and scores being visible via REST.
    # Retry up to 3 times with 15s sleeps; each attempt expects more scores than the last.
    captured = 0
    scores: list[dict] = []
    expected_min = 1  # at minimum one score should land per (item × metric); we run with item-limit and ~6 metrics
    for attempt in range(1, 4):
        time.sleep(15 if attempt == 1 else 10)
        scores = fetch_scores_window(env_vars, started, finished)
        if len(scores) >= expected_min:
            log.info(f"Score fetch attempt {attempt}: got {len(scores)} scores")
            break
        log.warning(f"Score fetch attempt {attempt}: got {len(scores)} scores (waiting for Langfuse propagation)")
    captured = append_scores(variant["code"], rep, scores, started, finished)
    log.info(f"Captured {captured} scores for variant {variant['code']} rep {rep:02d}")
    return code, captured


def main() -> int:
    p = argparse.ArgumentParser(description="RQ3 Phase 2 runner")
    p.add_argument("--dry-run", action="store_true", help="Print docker commands, don't run")
    p.add_argument("--variants", nargs="+", default=None, help="Subset of variant codes to run (e.g. B M)")
    p.add_argument("--reps", type=int, default=None, help="Repetitions per variant (default: from config)")
    p.add_argument("--item-limit", type=int, default=None, help="Override config.phase2.item_limit (smoke tests)")
    args = p.parse_args()

    env_vars = load_env_vars()
    cfg = load_config()
    if args.item_limit is not None:
        cfg["_item_limit_override"] = args.item_limit
        log.info(f"item_limit override (CLI): {args.item_limit}")
    ensure_results_dir()

    chosen = cfg["phase2"]["variants"]
    if args.variants:
        chosen = [v for v in chosen if v["code"] in args.variants]
        if not chosen:
            log.error(f"No matching variants for {args.variants}")
            return 2
    n_reps = args.reps if args.reps is not None else cfg["phase2"]["repetitions_per_variant"]

    log.info(f"Variants to run: {[v['code'] for v in chosen]}")
    log.info(f"Repetitions per variant: {n_reps}")
    log.info(f"Dataset run: {cfg['dataset_run_name']}")
    log.info(f"Items per run: {cfg['phase2']['item_limit']}")
    log.info(f"Image: {cfg['phase2']['image_tag']}")

    failures = 0
    for v in chosen:
        for rep in range(1, n_reps + 1):
            code, captured = run_one_variant_rep(v, rep, cfg, env_vars, args.dry_run)
            if code != 0:
                failures += 1
                log.error(f"Variant {v['code']} rep {rep} FAILED (exit={code}). Continuing.")
            # Small breather between containers (Langfuse propagation, rate limit headroom)
            time.sleep(2)

    log.info(f"Done. {failures} container failures across {len(chosen) * n_reps} runs.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
