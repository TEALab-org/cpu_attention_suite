"""Shared helpers for reading benchmark.py results (used by plot_results.py and make_table.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

# Fixed algorithm order and display names, so every figure and table lists them identically.
ALGORITHMS = [
    ("standard", "Standard"),
    ("streamed", "Streamed"),
    ("fa2", "FlashAttention-2"),
    ("summary", "ParallelSummary"),
    ("query", "QueryParallel"),
    ("hybrid", "Hybrid"),
    ("owned", "ParallelOwned"),  # Backward only (owned_bwd)
]

# Backward solvers named as in the paper; others share their forward counterpart's name
BACKWARD_NAMES = {
    "summary_bwd": "ParallelDenseBackward",
    "owned_bwd": "ParallelOwnedBackward",
}

ARCHITECTURES = {
    "skx": "Skylake",
    "icx": "Ice Lake",
    "spr": "Sapphire Rapids",
}

PAPI_EVENTS = [
    ("PAPI_L1_DCM", "L1"),
    ("PAPI_L2_DCM", "L2"),
    ("PAPI_L3_TCM", "L3"),
]


PASSES = ("forward", "backward")


def result_kind(kind: str, pass_name: str) -> str:
    """Result directory for a phase, matching benchmark.py: sweep/ or sweep_backward/, etc."""
    return kind if pass_name == "forward" else f"{kind}_{pass_name}"


def algorithm_key(base: str, pass_name: str) -> str:
    """Solver name for an algorithm in a pass: fa2 -> fa2_bwd for the backward pass."""
    return base if pass_name == "forward" else f"{base}_bwd"


def display_name(key: str) -> str:
    """Display name shared by an algorithm's forward and backward solvers (or its paper name)."""
    if key in BACKWARD_NAMES:
        return BACKWARD_NAMES[key]
    base = key[: -len("_bwd")] if key.endswith("_bwd") else key
    return dict(ALGORITHMS).get(base, key)


def load_records(directory: Path, include_partial: bool) -> List[dict]:
    """Experiment records in a tune/sweep/papi directory; finished ones only unless include_partial."""
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        status = rec.get("status")
        if status == "complete" or (include_partial and status in ("partial", "running") and rec.get("times")):
            records.append(rec)
    return records


def machine_threads(results_dir: Path, label: str) -> Optional[int]:
    paths = sorted((results_dir / label / "sweep").glob("*.json"))
    paths += sorted((results_dir / label / result_kind("sweep", "backward")).glob("*.json"))
    for path in paths:
        try:
            with open(path) as f:
                threads = json.load(f).get("threads")
        except (OSError, json.JSONDecodeError):
            continue
        if threads:
            return threads
    return None


def machine_title(results_dir: Path, label: str) -> str:
    name = ARCHITECTURES.get(label, label)
    threads = machine_threads(results_dir, label)
    return f"{name} ({threads} cores)" if threads else name
