#!/usr/bin/env python3
"""Plot dense forward-pass benchmark results across architectures.

Reads the per-experiment JSON files written by benchmark.py and draws a 4 x A grid
(A = one column per architecture label):

  row 1  runtime (median of repetitions, whiskers = min/max) over N
  row 2  L1 data cache misses over N   (PAPI sweep)
  row 3  L2 data cache misses over N
  row 4  L3 total cache misses over N

With --normalize every metric is divided by N^2 (per query-key pair), which removes the
quadratic trend so algorithms with similar scaling separate visually.

Rows share a y scale so architectures can be compared directly. Panels with no data yet
(e.g. before the PAPI phase has run) are marked as such rather than left blank.

Example:
    python python_scripts/plot_results.py \\
        --results-dir ../cpu_attention_results --output-dir ../cpu_attention_output
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

# Fixed algorithm order -> fixed colour/marker, so a series keeps its identity in every panel
# and in every figure regardless of which algorithms have data.
ALGORITHMS = [
    ("standard", "Standard"),
    ("streamed", "Streamed"),
    ("fa2", "FlashAttention-2"),
    ("summary", "ParallelSummary"),
    ("query", "QueryParallel"),
    ("hybrid", "Hybrid"),
]
# Validated categorical palette, slots 1-6 in order (blue, orange, aqua, yellow, magenta, green).
# Markers are a secondary encoding: six crossing lines need more than hue alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

ARCHITECTURES = {
    "skx": "Skylake",
    "icx": "Ice Lake",
    "spr": "Sapphire Rapids",
}

ROWS = [
    # (kind, metric key, y label)
    ("sweep", "time", "Runtime (s)"),
    ("papi", "PAPI_L1_DCM", "L1 data cache misses"),
    ("papi", "PAPI_L2_DCM", "L2 data cache misses"),
    ("papi", "PAPI_L3_TCM", "L3 cache misses"),
]

TEXT = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3dd"


@dataclass
class Series:
    n: List[int]
    median: List[float]
    low: List[float]
    high: List[float]
    partial: List[bool]


def load_records(directory: Path, include_partial: bool) -> List[dict]:
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


def metric_values(rec: dict, metric: str, normalize: bool) -> List[float]:
    if metric == "time":
        # ms -> s, or ms -> ns per query-key pair
        scale = 1e6 / rec["N"] ** 2 if normalize else 1e-3
        return [t * scale for t in rec.get("times", [])]
    scale = 1.0 / rec["N"] ** 2 if normalize else 1.0
    return [float(c[metric]) * scale for c in rec.get("counters", []) if metric in c]


def collect(records: List[dict], algorithm: str, metric: str, normalize: bool) -> Optional[Series]:
    points: Dict[int, Tuple[List[float], bool]] = {}
    for rec in records:
        if rec.get("name") != algorithm:
            continue
        values = metric_values(rec, metric, normalize)
        if values:
            points[rec["N"]] = (values, rec.get("status") != "complete")
    if not points:
        return None
    ns = sorted(points)
    vals = [points[n][0] for n in ns]
    return Series(
        n=ns,
        median=[statistics.median(v) for v in vals],
        low=[min(v) for v in vals],
        high=[max(v) for v in vals],
        partial=[points[n][1] for n in ns],
    )


def machine_title(results_dir: Path, label: str) -> str:
    name = ARCHITECTURES.get(label, label)
    threads = None
    for path in sorted((results_dir / label / "sweep").glob("*.json")):
        try:
            with open(path) as f:
                threads = json.load(f).get("threads")
        except (OSError, json.JSONDecodeError):
            continue
        if threads:
            break
    return f"{name} ({threads} cores)" if threads else name


def power_of_two_label(x: float, _pos=None) -> str:
    exp = round(float(x)).bit_length() - 1
    return rf"$2^{{{exp}}}$" if x > 0 and 2**exp == round(x) else ""


def style_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=8)


def plot(results_dir: Path, labels: List[str], include_partial: bool, usetex: bool, normalize: bool):
    if usetex:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Computer Modern Serif"]})
    else:
        plt.rcParams.update({"font.family": "serif"})

    data = {}
    all_n = set()
    for label in labels:
        for kind in {kind for kind, _, _ in ROWS}:
            records = load_records(results_dir / label / kind, include_partial)
            data[(label, kind)] = records
            all_n.update(r["N"] for r in records)

    n_rows, n_cols = len(ROWS), len(labels)
    fig = plt.figure(figsize=(3.6 * n_cols, 2.6 * n_rows + 0.8))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.18, wspace=0.12, top=0.90, bottom=0.07, left=0.08, right=0.98)

    axes = [[None] * n_cols for _ in range(n_rows)]
    for r in range(n_rows):
        for c in range(n_cols):
            share_y = axes[r][0] if c else None
            share_x = axes[0][c] if r else None
            axes[r][c] = fig.add_subplot(gs[r, c], sharex=share_x, sharey=share_y)

    handles = {}
    for r, (kind, metric, ylabel) in enumerate(ROWS):
        row_has_data = False
        for c, label in enumerate(labels):
            ax = axes[r][c]
            style_axis(ax)
            plotted = False
            for slot, (algorithm, display) in enumerate(ALGORITHMS):
                series = collect(data[(label, kind)], algorithm, metric, normalize)
                if series is None:
                    continue
                plotted = True
                err = [[m - lo for m, lo in zip(series.median, series.low)],
                       [hi - m for m, hi in zip(series.median, series.high)]]
                line = ax.errorbar(series.n, series.median, yerr=err, color=COLORS[slot],
                                   marker=MARKERS[slot], markersize=4.5, linewidth=1.6,
                                   elinewidth=0.9, capsize=2, label=display,
                                   markeredgecolor="white", markeredgewidth=0.6)
                if any(series.partial):
                    # Hollow markers flag points with fewer repetitions than requested
                    pn = [n for n, p in zip(series.n, series.partial) if p]
                    pm = [m for m, p in zip(series.median, series.partial) if p]
                    ax.plot(pn, pm, linestyle="none", marker=MARKERS[slot], markersize=6,
                            markerfacecolor="white", markeredgecolor=COLORS[slot], markeredgewidth=1.2)
                handles.setdefault(algorithm, line)
            if not plotted:
                ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center",
                        color=MUTED, fontsize=9, style="italic")
            row_has_data |= plotted
            if r == 0:
                ax.set_title(machine_title(results_dir, label), color=TEXT, fontsize=11)
            if c == 0:
                if normalize:
                    ylabel = "Time per pair (ns)" if metric == "time" else ylabel.replace("misses", r"misses / $N^2$")
                ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
            else:
                # which="both": log axes spanning under a decade label their minor ticks too
                ax.tick_params(axis="y", which="both", labelleft=False)
            if r == n_rows - 1:
                ax.set_xlabel(r"Sequence length $N$", color=TEXT, fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
        if not row_has_data:
            axes[r][0].set_yscale("linear")
            axes[r][0].set_yticks([])

    if all_n:
        ticks = sorted(all_n)
        for c in range(n_cols):
            ax = axes[0][c]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(power_of_two_label))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(min(ticks) / 1.3, max(ticks) * 1.3)

    ordered = [handles[a] for a, _ in ALGORITHMS if a in handles]
    if ordered:
        fig.legend(ordered, [h.get_label() for h in ordered], loc="upper center", ncol=len(ordered),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.985), labelcolor=TEXT)
    fig.text(0.5, 0.935, r"Dense forward pass, $d = 64$" + (" (hollow markers: incomplete repetitions)"
             if include_partial else ""), ha="center", color=MUTED, fontsize=9)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="benchmark.py --results-dir root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write figures into")
    parser.add_argument("--labels", default="skx,icx,spr",
                        help="Comma-separated machine labels, one column each (default: %(default)s)")
    parser.add_argument("--name", default="dense_forward", help="Output file stem (default: %(default)s)")
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated formats (default: %(default)s)")
    parser.add_argument("--include-partial", action="store_true",
                        help="Also plot experiments that have not finished all repetitions (hollow markers)")
    parser.add_argument("--normalize", action="store_true",
                        help="Divide every metric by N^2 (per query-key pair); default stem gets a _normalized suffix")
    parser.add_argument("--usetex", action="store_true", help="Render text with LaTeX (Computer Modern)")
    args = parser.parse_args()

    labels = [label for label in args.labels.split(",") if label]
    missing = [label for label in labels if not (args.results_dir / label).is_dir()]
    if missing:
        print(f"Warning: no results for {', '.join(missing)} under {args.results_dir}", file=sys.stderr)

    fig = plot(args.results_dir, labels, args.include_partial, args.usetex, args.normalize)
    name = args.name + ("_normalized" if args.normalize else "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats.split(","):
        out = args.output_dir / f"{name}.{fmt}"
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
