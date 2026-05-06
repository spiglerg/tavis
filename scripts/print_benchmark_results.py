#!/usr/bin/env python3
"""Print ASCII tables summarising the JSON output of `eval_benchmark.py`.

Each call to ``eval_benchmark.py`` writes one JSON file per
``(task, eval_mode)`` combination under ``--output/<checkpoint-name>/``
(default: ``results/<checkpoint-name>/``). This script aggregates those
JSONs into per-task / per-eval-mode success-rate tables with Wilson
95 % confidence intervals and a per-condition episode count.

Usage
-----
::

    # Single checkpoint:
    python scripts/print_benchmark_results.py results/pi0-tavis-head-gr1t2-headcam

    # Side-by-side comparison of multiple runs:
    python scripts/print_benchmark_results.py results/run_A results/run_B

The expected per-JSON structure (produced by eval_benchmark.py)::

    {
      "meta": {"task": "...", "eval_mode": "id", "model": "...", "camera": "...", "robot": "..."},
      "episodes": [{"success": True/False, ...}, ...]
    }
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ── Wilson score confidence interval for a proportion ────────────────────────
def wilson_ci(n_success: int, n_total: int, z: float = 1.96):
    """Return (lower, upper) 95 % CI for a proportion using Wilson score."""
    if n_total == 0:
        return (0.0, 0.0)
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z / denom * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Load results from a checkpoint directory ─────────────────────────────────
def load_checkpoint_results(ckpt_dir: Path) -> dict:
    """Load all JSON result files from a checkpoint directory.

    Returns a dict keyed by ``(task, eval_mode)`` → parsed JSON.
    """
    results = {}
    for jf in sorted(ckpt_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        meta = data["meta"]
        key = (meta["task"], meta["eval_mode"])
        results[key] = data
    return results


# ── Formatting helpers ───────────────────────────────────────────────────────
def fmt_sr(n_success, n_total, width=22):
    """Format success rate with CI: '73.0% [63.5, 81.0]'."""
    if n_total == 0:
        return "-".center(width)
    sr = n_success / n_total * 100
    lo, hi = wilson_ci(n_success, n_total)
    return f"{sr:5.1f}% [{lo*100:4.1f}, {hi*100:4.1f}]".center(width)


def make_table(headers, rows, col_widths=None):
    """Build a simple ASCII table string."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h) + 2
            for row in rows:
                if i < len(row):
                    w = max(w, len(str(row[i])) + 2)
            col_widths.append(w)

    lines = []
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    lines.append(sep)
    lines.append("|" + "|".join(str(h).center(w) for h, w in zip(headers, col_widths)) + "|")
    lines.append(sep)
    for row in rows:
        cells = []
        for i, w in enumerate(col_widths):
            val = row[i] if i < len(row) else ""
            cells.append(str(val).center(w))
        lines.append("|" + "|".join(cells) + "|")
    lines.append(sep)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Print ASCII success-rate tables from eval_benchmark.py JSON output"
    )
    parser.add_argument("dirs", nargs="*",
                        help="One or more checkpoint result directories (each contains the "
                             "per-(task, eval-mode) JSONs written by eval_benchmark.py)")
    args = parser.parse_args()

    if not args.dirs:
        print(
            "Usage:\n"
            "    python scripts/print_benchmark_results.py <results_dir> [<results_dir> ...]\n"
            "\n"
            "Each <results_dir> is the directory eval_benchmark.py wrote into\n"
            "(named after the checkpoint, e.g. results/pi0-tavis-head-gr1t2-headcam/).\n"
            "\n"
            "Example:\n"
            "    python scripts/print_benchmark_results.py \\\n"
            "        results/pi0-tavis-head-gr1t2-headcam\n"
            "\n"
            "    # Side-by-side comparison of multiple runs:\n"
            "    python scripts/print_benchmark_results.py \\\n"
            "        results/run_A results/run_B",
            file=sys.stderr,
        )
        sys.exit(1)

    ckpt_dirs = [Path(d) for d in args.dirs]
    for d in ckpt_dirs:
        if not d.is_dir():
            print(f"Not a directory: {d}", file=sys.stderr)
            sys.exit(1)

    # Canonical task order (matches the suite definitions in tavis.benchmark.suites).
    # Unknown tasks are appended in the order they first appear.
    TASK_ORDER = [
        # TAVIS-HEAD
        "clutter_pick_lift",
        "clutter_pick_cube",
        "conditional_pick",
        "wait_then_act",
        "multi_shelf_scan",
        # TAVIS-HANDS
        "peeking_box",
        "occluded_reach",
        "blocked_clutter_pick_cube",
    ]
    EVAL_MODES = ["id", "ood_spatial"]

    for ckpt_dir in ckpt_dirs:
        results = load_checkpoint_results(ckpt_dir)
        if not results:
            print(f"\n  (no JSON files in {ckpt_dir})\n")
            continue

        first = next(iter(results.values()))
        meta = first["meta"]
        ckpt_label = ckpt_dir.name
        model_label = (
            f"{meta.get('model', '?')} / {meta.get('camera', '?')} / {meta.get('robot', '?')}"
        )

        # Tasks and eval modes actually present (preserving canonical order)
        tasks_present = [t for t in TASK_ORDER if any(k[0] == t for k in results)]
        for (t, _) in results:
            if t not in tasks_present:
                tasks_present.append(t)
        modes_present = [m for m in EVAL_MODES if any(k[1] == m for k in results)]
        for (_, m) in results:
            if m not in modes_present:
                modes_present.append(m)

        print()
        print("=" * 70)
        print(f"  Checkpoint: {ckpt_label}")
        print(f"  Config:     {model_label}")
        print("=" * 70)

        # ── Per-task × per-eval-mode success rates with CIs ─────────────────
        print("\n  Success Rate by Task and Eval Mode")
        print("  " + "-" * 36)

        headers = ["Task"] + list(modes_present) + (["delta (id − ood)"] if {"id", "ood_spatial"} <= set(modes_present) else [])
        rows = []
        mode_totals = {m: (0, 0) for m in modes_present}

        for task in tasks_present:
            row = [task]
            task_srs = {}
            for mode in modes_present:
                key = (task, mode)
                if key in results:
                    eps = results[key]["episodes"]
                    ns = sum(1 for e in eps if e["success"])
                    nt = len(eps)
                    row.append(fmt_sr(ns, nt))
                    task_srs[mode] = ns / nt if nt > 0 else None
                    s, t = mode_totals[mode]
                    mode_totals[mode] = (s + ns, t + nt)
                else:
                    row.append("-".center(22))
                    task_srs[mode] = None

            if {"id", "ood_spatial"} <= set(modes_present):
                if task_srs.get("id") is not None and task_srs.get("ood_spatial") is not None:
                    d = (task_srs["id"] - task_srs["ood_spatial"]) * 100
                    row.append(f"{d:+.1f}pp".center(18))
                else:
                    row.append("-".center(18))
            rows.append(row)

        # Average row
        rows.append([""] * len(headers))
        avg_row = ["AVERAGE"]
        avg_srs = {}
        for mode in modes_present:
            s, t = mode_totals[mode]
            avg_row.append(fmt_sr(s, t))
            avg_srs[mode] = s / t if t > 0 else None
        if {"id", "ood_spatial"} <= set(modes_present):
            if avg_srs.get("id") is not None and avg_srs.get("ood_spatial") is not None:
                d = (avg_srs["id"] - avg_srs["ood_spatial"]) * 100
                avg_row.append(f"{d:+.1f}pp".center(18))
            else:
                avg_row.append("-".center(18))
        rows.append(avg_row)

        print(make_table(headers, rows))

        # ── Episode counts per condition ────────────────────────────────────
        print("\n  Episodes per condition:")
        for task in tasks_present:
            parts = []
            for mode in modes_present:
                key = (task, mode)
                if key in results:
                    n = len(results[key]["episodes"])
                    parts.append(f"{mode}={n}")
            print(f"    {task}: {', '.join(parts)}")
        print()


if __name__ == "__main__":
    main()
