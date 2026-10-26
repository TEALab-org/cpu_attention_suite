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
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_io import (  # noqa: E402
    ALGORITHMS, PASSES, algorithm_key, display_name, load_records, machine_title, result_kind,
)

# Each algorithm keeps a fixed colour/marker (by its slot in results_io.ALGORITHMS) in every
# panel and figure, regardless of which algorithms have data; a backward solver (fa2_bwd)
# shares its forward counterpart's slot.
# Validated categorical palette, slots 1-7 in order (blue, orange, aqua, yellow, magenta, green,
# violet). Markers are a secondary encoding: several crossing lines need more than hue alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

ROWS = [
    # (kind, metric key, y label)
    ("sweep", "time", "Runtime (s)"),
    ("papi", "PAPI_L1_DCM", "L1 data cache misses"),
    ("papi", "PAPI_L2_DCM", "L2 data cache misses"),
    ("papi", "PAPI_L3_TCM", "L3 cache misses"),
]

# Default series per pass; None draws every algorithm that has results
DEFAULT_ALGORITHMS = {
    "forward": ["fa2", "summary", "query", "hybrid"],
    "backward": None,
}

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


def plot(results_dir: Path, labels: List[str], include_partial: bool, usetex: bool, normalize: bool,
         pass_name: str, algorithms: Optional[List[str]] = None):
    """algorithms: solver names to draw (e.g. fa2, summary_bwd); None draws every algorithm with data."""
    if usetex:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Computer Modern Serif"]})
    else:
        plt.rcParams.update({"font.family": "serif"})

    data = {}
    all_n = set()
    for label in labels:
        for kind in {kind for kind, _, _ in ROWS}:
            records = load_records(results_dir / label / result_kind(kind, pass_name), include_partial)
            data[(label, kind)] = records
            all_n.update(r["N"] for r in records)

    n_rows, n_cols = len(ROWS), len(labels)
    fig = plt.figure(figsize=(3.6 * n_cols, 2.6 * n_rows + 0.8))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.18, wspace=0.12, top=0.90, bottom=0.07,
                          left=min(0.24, 0.24 / n_cols), right=0.98)  # Same absolute margin for any column count

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
            for slot, (base, _) in enumerate(ALGORITHMS):
                algorithm = algorithm_key(base, pass_name)
                if algorithms is not None and algorithm not in algorithms:
                    continue  # Skipped series keep their slot, so colours stay fixed
                series = collect(data[(label, kind)], algorithm, metric, normalize)
                if series is None:
                    continue
                plotted = True
                err = [[m - lo for m, lo in zip(series.median, series.low)],
                       [hi - m for m, hi in zip(series.median, series.high)]]
                line = ax.errorbar(series.n, series.median, yerr=err, color=COLORS[slot],
                                   marker=MARKERS[slot], markersize=4.5, linewidth=1.6,
                                   elinewidth=0.9, capsize=2, label=display_name(algorithm),
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

    ordered = [handles[algorithm_key(a, pass_name)] for a, _ in ALGORITHMS if algorithm_key(a, pass_name) in handles]
    # One legend row if it fits; otherwise wrap onto as few rows as fit the figure width, and move
    # the subtitle and panels below the legend's measured bottom edge
    subtitle_y = 0.935
    if ordered:
        renderer = fig.canvas.get_renderer()
        legend = None
        for ncol in range(len(ordered), 0, -1):
            if legend is not None:
                legend.remove()
            legend = fig.legend(ordered, [h.get_label() for h in ordered], loc="upper center", ncol=ncol,
                                frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.985), labelcolor=TEXT)
            if legend.get_window_extent(renderer).width <= 0.98 * fig.bbox.width:
                break
        if ncol < len(ordered):
            subtitle_y = legend.get_window_extent(renderer).y0 / fig.bbox.height - 0.012
            gs.update(top=subtitle_y - 0.035)
    fig.text(0.5, subtitle_y, rf"Dense {pass_name} pass, $d = 64$" + (" (hollow markers: incomplete repetitions)"
             if include_partial else ""), ha="center", color=MUTED, fontsize=9)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="benchmark.py --results-dir root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write figures into")
    parser.add_argument("--labels", default="skx,icx,spr",
                        help="Comma-separated machine labels, one column each (default: %(default)s)")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated solvers to draw (default: forward fa2,summary,query,hybrid; "
                             "backward all with data)")
    parser.add_argument("--pass", dest="pass_name", choices=PASSES, default="forward",
                        help="Plot forward or backward (*_bwd) results (default: %(default)s)")
    parser.add_argument("--name", default=None, help="Output file stem (default: dense_<pass>)")
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

    if args.algorithms:
        algorithms = [a for a in args.algorithms.split(",") if a]
    else:
        algorithms = DEFAULT_ALGORITHMS[args.pass_name]
    fig = plot(args.results_dir, labels, args.include_partial, args.usetex, args.normalize, args.pass_name,
               algorithms)
    name = (args.name or f"dense_{args.pass_name}") + ("_normalized" if args.normalize else "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats.split(","):
        out = args.output_dir / f"{name}.{fmt}"
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
