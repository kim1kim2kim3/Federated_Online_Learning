#!/usr/bin/env python3
"""Run a full-round baseline with streaming metric output.

This avoids the interactive runner's verbose per-client log and writes CSV rows
as each round finishes, so partial progress survives if a long run is killed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fl-server"))

from Fedostc import FedOSTC  # noqa: E402
from Refol import REFOL  # noqa: E402

SERVER_REGISTRY = {
    "refol": REFOL,
    "fedostc": FedOSTC,
}


def tensor_scalar(value: Any) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    return float(value)


def read_rss_mb() -> float:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def cuda_memory_stats_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 3),
        "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024.0 * 1024.0), 3),
        "cuda_max_allocated_mb": round(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            3,
        ),
    }


def write_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def load_config(path: Path, overrides: dict[str, Any]) -> SimpleNamespace:
    docs = list(yaml.load_all(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader))
    cfg = dict(docs[0])
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    return SimpleNamespace(**cfg)


def summarize_csv(path: Path) -> dict[str, Any]:
    count = 0
    sum_rmse = 0.0
    sum_mae = 0.0
    last: dict[str, float] | None = None
    best_rmse: dict[str, float] | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count += 1
            current = {
                "round": int(row["round"]),
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"]),
            }
            sum_rmse += current["rmse"]
            sum_mae += current["mae"]
            last = current
            if best_rmse is None or current["rmse"] < best_rmse["rmse"]:
                best_rmse = current
    return {
        "rounds_completed": count,
        "mean_rmse": sum_rmse / count if count else None,
        "mean_mae": sum_mae / count if count else None,
        "last": last,
        "best_rmse": best_rmse,
    }


def maybe_write_combined_summary(output_dir: Path) -> None:
    summaries = {}
    for name in SERVER_REGISTRY:
        summary_path = output_dir / f"{name}_summary.json"
        if not summary_path.exists():
            return
        summaries[name] = json.loads(summary_path.read_text(encoding="utf-8"))
    (output_dir / "combined_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg_model", required=True, choices=sorted(SERVER_REGISTRY))
    parser.add_argument("--config_path", default="default_config.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument(
        "--cuda_memory_guard_mb",
        type=float,
        default=0.0,
        help=(
            "Stop after the current round if CUDA reserved memory exceeds this "
            "threshold. 0 disables the guard."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.agg_model}_metrics.csv"
    progress_path = output_dir / f"{args.agg_model}_progress.log"
    summary_path = output_dir / f"{args.agg_model}_summary.json"

    cfg = load_config(
        ROOT / args.config_path,
        {
            "agg_model": args.agg_model,
            "rounds": args.rounds,
        },
    )
    server = SERVER_REGISTRY[args.agg_model](cfg)

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            server.boot()

    rounds = server._resolve_rounds()
    start_time = time.time()
    aborted: dict[str, Any] | None = None

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_handle, progress_path.open(
        "w", encoding="utf-8"
    ) as progress_handle, open(os.devnull, "w", encoding="utf-8") as devnull:
        writer = csv.writer(metrics_handle)
        writer.writerow(["round", "rmse", "mae"])
        metrics_handle.flush()
        write_jsonl(
            progress_handle,
            {
                "event": "start",
                "agg_model": args.agg_model,
                "rounds": rounds,
                "rss_mb": read_rss_mb(),
                **cuda_memory_stats_mb(),
            },
        )

        for rround in range(1, rounds + 1):
            with contextlib.redirect_stdout(devnull):
                train_log = server.train_round(rround)
            log = train_log["log"]
            writer.writerow([rround, tensor_scalar(log["rmse"]), tensor_scalar(log["mae"])])

            cuda_stats = cuda_memory_stats_mb()
            if rround % args.progress_every == 0 or rround == rounds:
                elapsed = time.time() - start_time
                write_jsonl(
                    progress_handle,
                    {
                        "event": "progress",
                        "round": rround,
                        "rounds": rounds,
                        "elapsed_sec": round(elapsed, 3),
                        "sec_per_round": round(elapsed / rround, 6),
                        "rss_mb": round(read_rss_mb(), 3),
                        **cuda_stats,
                    },
                )
                metrics_handle.flush()
                gc.collect()

            if (
                args.cuda_memory_guard_mb > 0
                and cuda_stats
                and cuda_stats["cuda_reserved_mb"] > args.cuda_memory_guard_mb
            ):
                elapsed = time.time() - start_time
                aborted = {
                    "aborted": True,
                    "abort_reason": "cuda_memory_guard_exceeded",
                    "round": rround,
                    "cuda_memory_guard_mb": args.cuda_memory_guard_mb,
                    **cuda_stats,
                }
                write_jsonl(
                    progress_handle,
                    {
                        "event": "abort",
                        "elapsed_sec": round(elapsed, 3),
                        **aborted,
                    },
                )
                metrics_handle.flush()
                break

        if aborted is None and hasattr(server, "flush"):
            with contextlib.redirect_stdout(devnull):
                server.flush(rounds)

    summary = summarize_csv(metrics_path)
    summary.update(
        {
            "agg_model": args.agg_model,
            "metrics_path": str(metrics_path),
            "progress_path": str(progress_path),
            "elapsed_sec": round(time.time() - start_time, 3),
            "rss_mb": round(read_rss_mb(), 3),
            **cuda_memory_stats_mb(),
        }
    )
    if aborted is not None:
        summary.update(aborted)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    maybe_write_combined_summary(output_dir)
    print(json.dumps(summary, indent=2))
    return 2 if aborted is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
