#!/usr/bin/env python3
"""Plot the sparsity experiment: speedup of the sparse solver as one kind of sparsity grows.

Reads the sparsity/ (or sparsity_backward/) results written by benchmark.py --phases sparsity and
draws one panel per architecture label. Each kind of sparsity (block-random mask, banded mask,
zero Q rows, zero K/V rows, zero dO rows) is one series:

  x  the achieved work reduction 1 / (fraction of the work left): 1 / mask density for masks,
     N / (active rows) for zero rows. Ticks are labelled with the sparsity level s = 1 - 1/x.
  y  speedup over the baseline, median of repetitions (whiskers = min/max)

The dashed diagonal y = x is the ideal speedup, where the time falls in proportion to the work.

The baseline is the fastest dense solver of the same pass at the same N, from that machine's
sweep (--baseline dense, the default); with --baseline self, or when the dense sweep has no
result at that N, it is the sparse solver itself with nothing sparse.

Example:
    python python_scripts/plot_sparsity.py \\
        --results-dir ../cpu_attention_results --output-dir ../cpu_attention_output
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import COLORS, GRID, MARKERS, MUTED, TEXT  # noqa: E402
from results_io import PASSES, display_name, load_records, machine_title, result_kind  # noqa: E402

# Fixed order, colour and marker per kind of sparsity, whatever has data
KINDS = [
    ("block", "Block-random mask"),
    ("band", "Banded mask"),
    ("element", "Element-random mask"),
    ("zero_q", r"Zero $Q$ rows"),
    ("zero_kv", r"Zero $K$/$V$ rows"),
    ("zero_do", r"Zero $dO$ rows"),
]
ZERO_ROW_FIELDS = {"zero_q": "zero_q_rows", "zero_kv": "zero_kv_rows", "zero_do": "zero_do_rows"}
LEVEL_TICKS = [0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def work_reduction(rec: dict) -> Optional[float]:
    """1 / (fraction of the work left), from the achieved mask density or zero-row count."""
    kind = rec.get("sparsity_type")
    if kind == "none":
        return 1.0
    achieved = rec.get("sparsity")
    if not achieved:
        return None
    if kind in ZERO_ROW_FIELDS:
        active = rec["N"] - achieved[ZERO_ROW_FIELDS[kind]]
        return rec["N"] / active if active > 0 else None
    density = achieved.get("mask_density")
    return 1.0 / density if density else None


def dense_baseline(results_dir: Path, label: str, pass_name: str, n: int,
                   include_partial: bool) -> Optional[Tuple[str, float]]:
    """(solver, median ms) of the fastest dense solver at N in this machine's sweep."""
    best = None
    for rec in load_records(results_dir / label / result_kind("sweep", pass_name), include_partial):
        if rec["N"] != n or rec["name"].startswith("sparse_") or not rec.get("times"):
            continue
        median = statistics.median(rec["times"])
        if best is None or median < best[1]:
            best = (rec["name"], median)
    return best


def style_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=8)


def level_label(level: float) -> str:
    return f"{100 * level:g}\\%" if plt.rcParams["text.usetex"] else f"{100 * level:g}%"


def set_level_ticks(ax, max_x: float) -> None:
    """x ticks at round sparsity levels up to the largest work reduction plotted."""
    ticks = [t for t in LEVEL_TICKS if 1 / (1 - t) <= max_x * 1.5]
    ax.set_xlim(1 / 1.25, max_x * 1.3)
    ax.xaxis.set_major_locator(FixedLocator([1 / (1 - t) for t in ticks]))
    ax.xaxis.set_major_formatter(plt.FixedFormatter([level_label(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def set_style(usetex: bool) -> None:
    if usetex:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Computer Modern Serif"]})
    else:
        plt.rcParams.update({"font.family": "serif"})


def sparsity_records(results_dir: Path, label: str, pass_name: str, include_partial: bool) -> List[dict]:
    return load_records(results_dir / label / result_kind("sparsity", pass_name), include_partial)


def add_legend_and_subtitle(fig, ordered: List, pass_name: str, sizes: set, include_partial: bool) -> None:
    """Legend in as many columns as fit the figure width (at most 4), then the subtitle below it."""
    renderer = fig.canvas.get_renderer()
    legend = None
    for ncol in range(min(len(ordered), 4), 0, -1):
        if legend is not None:
            legend.remove()
        legend = fig.legend(ordered, [h.get_label() for h in ordered], loc="upper center", ncol=ncol,
                            frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995), labelcolor=TEXT)
        if legend.get_window_extent(renderer).width <= 0.98 * fig.bbox.width:
            break
    if ncol < min(len(ordered), 4):  # Taller legend: move the subtitle and panels below it
        legend_bottom = legend.get_window_extent(renderer).y0 / fig.bbox.height
        fig.subplots_adjust(top=legend_bottom - 0.12)
        subtitle_y = legend_bottom - 0.05
    else:
        subtitle_y = 0.80
    size = ", ".join(rf"$N = 2^{{{n.bit_length() - 1}}}$" for n in sorted(sizes))
    fig.text(0.5, subtitle_y, f"Sparse {pass_name} pass, d = 64" + (f", {size}" if size else "")
             + (" (hollow markers: incomplete repetitions)" if include_partial else ""),
             ha="center", color=MUTED, fontsize=9)


def plot(results_dir: Path, labels: List[str], pass_name: str, baseline: str, include_partial: bool,
         usetex: bool):
    set_style(usetex)

    n_cols = len(labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 3.4), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(top=0.72, bottom=0.17, left=min(0.24, 0.24 / n_cols) / 1.6, right=0.98, wspace=0.12)

    handles: Dict[str, object] = {}
    empty = set()  # Panels without data get no ideal line through their "no data yet"
    max_x = 1.0
    sizes = set()
    for c, label in enumerate(labels):
        ax = axes[c]
        style_axis(ax)
        records = sparsity_records(results_dir, label, pass_name, include_partial)
        none = [r for r in records if r.get("sparsity_type") == "none"]
        note = None
        base_ms = None
        if records:
            n = records[0]["N"]
            sizes.add(n)
            dense = dense_baseline(results_dir, label, pass_name, n, include_partial) if baseline == "dense" else None
            if dense is not None:
                base_ms = dense[1]
                note = f"vs {display_name(dense[0])}"
            elif none:
                base_ms = statistics.median(none[0]["times"])
                note = "vs itself, nothing sparse"
        if base_ms is None:
            empty.add(c)
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center",
                    color=MUTED, fontsize=9, style="italic")
        else:
            for slot, (kind, name) in enumerate(KINDS):
                points = []
                for rec in none + [r for r in records if r.get("sparsity_type") == kind]:
                    x = work_reduction(rec)
                    if x is not None and rec.get("times"):
                        times = rec["times"]
                        points.append((x, base_ms / statistics.median(times), base_ms / max(times),
                                       base_ms / min(times), rec.get("status") != "complete"))
                if len(points) <= len(none):
                    continue  # Only the shared nothing-sparse point: this kind has not run here
                points.sort()
                xs, ys, lo, hi, partial = zip(*points)
                max_x = max(max_x, max(xs))
                line = ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                                   color=COLORS[slot], marker=MARKERS[slot], markersize=4.5, linewidth=1.6,
                                   elinewidth=0.9, capsize=2, label=name,
                                   markeredgecolor="white", markeredgewidth=0.6)
                if any(partial):
                    ax.plot([x for x, p in zip(xs, partial) if p], [y for y, p in zip(ys, partial) if p],
                            linestyle="none", marker=MARKERS[slot], markersize=6, markerfacecolor="white",
                            markeredgecolor=COLORS[slot], markeredgewidth=1.2)
                handles.setdefault(kind, line)
            ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=8)
        ax.set_title(machine_title(results_dir, label), color=TEXT, fontsize=11)
        ax.set_xlabel(r"Sparsity level $s$ (work reduction $1/(1-s)$)", color=TEXT, fontsize=9)
        if c == 0:
            ax.set_ylabel("Speedup", color=TEXT, fontsize=9)

    # Ideal speedup y = x, and ticks at round sparsity levels, on every panel
    for c, ax in enumerate(axes):
        if c not in empty:
            ax.plot([1, max_x * 1.3], [1, max_x * 1.3], color=MUTED, linestyle="--", linewidth=1.0, zorder=0)
        set_level_ticks(ax, max_x)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}"))
    ideal = plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.0, label="Ideal")

    ordered = [handles[k] for k, _ in KINDS if k in handles] + [ideal]
    add_legend_and_subtitle(fig, ordered, pass_name, sizes, include_partial)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="benchmark.py --results-dir root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write figures into")
    parser.add_argument("--labels", default="skx,icx,spr",
                        help="Comma-separated machine labels, one panel each (default: %(default)s)")
    parser.add_argument("--pass", dest="pass_name", choices=PASSES, default="forward",
                        help="Plot the sparse forward or backward results (default: %(default)s)")
    parser.add_argument("--baseline", choices=["dense", "self"], default="dense",
                        help="Speedup over the fastest dense solver at the same N, or over the sparse "
                             "solver with nothing sparse (default: %(default)s)")
    parser.add_argument("--name", default=None, help="Output file stem (default: sparsity_<pass>)")
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated formats (default: %(default)s)")
    parser.add_argument("--include-partial", action="store_true",
                        help="Also plot experiments that have not finished all repetitions (hollow markers)")
    parser.add_argument("--usetex", action="store_true", help="Render text with LaTeX (Computer Modern)")
    args = parser.parse_args()

    labels = [label for label in args.labels.split(",") if label]
    missing = [label for label in labels if not (args.results_dir / label).is_dir()]
    if missing:
        print(f"Warning: no results for {', '.join(missing)} under {args.results_dir}", file=sys.stderr)

    fig = plot(args.results_dir, labels, args.pass_name, args.baseline, args.include_partial, args.usetex)
    name = args.name or f"sparsity_{args.pass_name}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats.split(","):
        out = args.output_dir / f"{name}.{fmt}"
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
