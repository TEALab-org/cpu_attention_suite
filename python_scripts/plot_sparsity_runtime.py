#!/usr/bin/env python3
"""Plot the sparsity experiment as runtime: the sparse solver's time as one kind of sparsity grows.

The runtime counterpart of plot_sparsity.py, with the same panels (one per architecture label),
series (one per kind of sparsity) and x axis (the achieved work reduction, ticks labelled with the
sparsity level s):

  y  runtime in seconds, median of repetitions (whiskers = min/max)

Two reference lines per panel:
  dashed  ideal: the sparse solver's time with nothing sparse, scaled by the work left, (1 - s)
  dotted  the fastest dense solver of the same pass at the same N, from that machine's sweep
          (named in the panel's corner); left out when the sweep has no result at that N

Example:
    python python_scripts/plot_sparsity_runtime.py \\
        --results-dir ../cpu_attention_results --output-dir ../cpu_attention_output
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import COLORS, MARKERS, MUTED, TEXT  # noqa: E402
from plot_sparsity import (  # noqa: E402
    KINDS, add_legend_and_subtitle, dense_baseline, set_level_ticks, set_style, sparsity_records, style_axis,
    work_reduction,
)
from results_io import PASSES, display_name, machine_title  # noqa: E402


def plot(results_dir: Path, labels: List[str], pass_name: str, include_partial: bool, usetex: bool):
    set_style(usetex)

    n_cols = len(labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 3.4), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(top=0.72, bottom=0.17, left=min(0.24, 0.24 / n_cols) / 1.6, right=0.98, wspace=0.12)

    handles: Dict[str, object] = {}
    references = []  # (axis, nothing-sparse seconds, dense seconds or None) per panel with data
    max_x = 1.0
    sizes = set()
    for c, label in enumerate(labels):
        ax = axes[c]
        style_axis(ax)
        records = sparsity_records(results_dir, label, pass_name, include_partial)
        none = [r for r in records if r.get("sparsity_type") == "none"]
        if not records:
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center",
                    color=MUTED, fontsize=9, style="italic")
        else:
            n = records[0]["N"]
            sizes.add(n)
            for slot, (kind, name) in enumerate(KINDS):
                points = []
                for rec in none + [r for r in records if r.get("sparsity_type") == kind]:
                    x = work_reduction(rec)
                    if x is not None and rec.get("times"):
                        times = [t / 1000 for t in rec["times"]]
                        points.append((x, statistics.median(times), min(times), max(times),
                                       rec.get("status") != "complete"))
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
            dense = dense_baseline(results_dir, label, pass_name, n, include_partial)
            if dense is not None:
                # Lower left stays clear: the curves fall from the upper left to the lower right
                ax.text(0.03, 0.04, f"dense: {display_name(dense[0])}", transform=ax.transAxes,
                        ha="left", va="bottom", color=MUTED, fontsize=8)
            references.append((ax, statistics.median(none[0]["times"]) / 1000 if none else None,
                               dense[1] / 1000 if dense else None))
        ax.set_title(machine_title(results_dir, label), color=TEXT, fontsize=11)
        ax.set_xlabel(r"Sparsity level $s$ (work reduction $1/(1-s)$)", color=TEXT, fontsize=9)
        if c == 0:
            ax.set_ylabel("Runtime (s)", color=TEXT, fontsize=9)

    # Reference lines span the full x range, so they are drawn once max_x is known
    span = [1 / 1.25, max_x * 1.3]
    has_ideal = has_dense = False
    for ax, none_s, dense_s in references:
        if none_s is not None:
            ax.plot(span, [none_s / x for x in span], color=MUTED, linestyle="--", linewidth=1.0, zorder=0)
            has_ideal = True
        if dense_s is not None:
            ax.axhline(dense_s, color=MUTED, linestyle=":", linewidth=1.2, zorder=0)
            has_dense = True
    for ax in axes:
        set_level_ticks(ax, max_x)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}"))

    ordered = [handles[k] for k, _ in KINDS if k in handles]
    if has_ideal:
        ordered.append(plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.0, label="Ideal"))
    if has_dense:
        ordered.append(plt.Line2D([], [], color=MUTED, linestyle=":", linewidth=1.2,
                                  label="Fastest dense solver"))
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
    parser.add_argument("--name", default=None, help="Output file stem (default: sparsity_runtime_<pass>)")
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated formats (default: %(default)s)")
    parser.add_argument("--include-partial", action="store_true",
                        help="Also plot experiments that have not finished all repetitions (hollow markers)")
    parser.add_argument("--usetex", action="store_true", help="Render text with LaTeX (Computer Modern)")
    args = parser.parse_args()

    labels = [label for label in args.labels.split(",") if label]
    missing = [label for label in labels if not (args.results_dir / label).is_dir()]
    if missing:
        print(f"Warning: no results for {', '.join(missing)} under {args.results_dir}", file=sys.stderr)

    fig = plot(args.results_dir, labels, args.pass_name, args.include_partial, args.usetex)
    name = args.name or f"sparsity_runtime_{args.pass_name}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats.split(","):
        out = args.output_dir / f"{name}.{fmt}"
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
