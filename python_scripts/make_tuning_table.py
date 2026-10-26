#!/usr/bin/env python3
"""Generate a small LaTeX table of the parameters chosen by benchmark.py's tuning phase.

Rows are grouped by architecture, one row per algorithm, with the tuned base case b, grain g
and processor budget p read from <label>/tune/best.json (or tune_backward/best.json with
--pass backward). --times adds each winner's median runtime at the tuning size. The script also prints each algorithm's best and worst tuning configuration to
stdout, showing how much the choice mattered.

The table uses table (pass --wide for table*), with the caption below the tabular.
Requires \\usepackage{booktabs} and \\usepackage{multirow}.

Example:
    python python_scripts/make_tuning_table.py \\
        --results-dir ../cpu_attention_results --output-dir ../cpu_attention_output
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_table import DEFAULT_ALGORITHMS, format_number  # noqa: E402
from results_io import ARCHITECTURES, PASSES, display_name, machine_threads, result_kind  # noqa: E402

PARAM_ORDER = ["b", "g", "p"]


def load_tuning(results_dir: Path, label: str, pass_name: str) -> Tuple[Dict[str, dict], List[dict]]:
    """(best parameters per algorithm, every finished tuning record) for one machine."""
    tune_dir = results_dir / label / result_kind("tune", pass_name)
    best_path = tune_dir / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    records = []
    for path in sorted(tune_dir.glob("*.json")):
        if path.name == "best.json":
            continue
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("status") == "complete" and rec.get("times"):
            records.append(rec)
    return best, records


def median_ms(rec: dict) -> float:
    return statistics.median(rec["times"])


def format_params(params: dict) -> str:
    """Plain-text parameters for the console summary, e.g. "b=128 g=256"."""
    return " ".join(f"{k}={params[k]}" for k in PARAM_ORDER if k in params)


def build_table(results_dir: Path, labels: List[str], algorithms: List[str], pass_name: str, times: bool,
                caption: Optional[str], latex_label: str, wide: bool) -> Tuple[str, List[str], List[str]]:
    notes: List[str] = []
    spreads: List[str] = []
    tune_sizes = set()

    env = "table*" if wide else "table"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",  # Narrow gaps (and no outer padding) so it fits one column
        r"  \begin{tabular}{@{}ll" + "r" * (len(PARAM_ORDER) + (1 if times else 0)) + "@{}}",
        r"    \toprule",
        "    Architecture & Algorithm & " + " & ".join(rf"${k}$" for k in PARAM_ORDER)
        + (r" & Time (ms)" if times else "") + r" \\",
        r"    \midrule",
    ]

    for g, arch in enumerate(labels):
        best, records = load_tuning(results_dir, arch, pass_name)
        if not best:
            notes.append(f"{arch}: no {result_kind('tune', pass_name)}/best.json")

        threads = machine_threads(results_dir, arch)
        name = ARCHITECTURES.get(arch, arch)
        arch_cell = (rf"\multirow{{{len(algorithms)}}}{{*}}{{\begin{{tabular}}[c]{{@{{}}l@{{}}}}{name}\\"
                     rf"{threads} cores\end{{tabular}}}}" if threads else rf"\multirow{{{len(algorithms)}}}{{*}}{{{name}}}")

        for i, algorithm in enumerate(algorithms):
            mine = [r for r in records if r.get("name") == algorithm]
            tune_sizes.update(r["N"] for r in mine)
            params = best.get(algorithm)
            # "--" for parameters the algorithm does not take, or no tuning result at all
            cells = [str(params[k]) if params and k in params else "--" for k in PARAM_ORDER]
            if times:
                winner = next((r for r in mine if params is not None and r.get("params") == params), None)
                cells.append(format_number(median_ms(winner)) if winner else "--")
            lines.append("    " + " & ".join([arch_cell if i == 0 else "", display_name(algorithm)] + cells) + r" \\")

            if mine:
                ranked = sorted(mine, key=median_ms)
                lo, hi = ranked[0], ranked[-1]
                spreads.append(f"{arch:4} {algorithm:12} best {format_params(lo['params']):18} "
                               f"{median_ms(lo):7.0f} ms   worst {format_params(hi['params']):18} "
                               f"{median_ms(hi):8.0f} ms   ({median_ms(hi) / median_ms(lo):.1f}x, "
                               f"{len(mine)} configs)")
        if g + 1 < len(labels):
            lines.append(r"    \midrule")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption or default_caption(tune_sizes, pass_name, times)}}}",
        rf"  \label{{{latex_label}}}",
        rf"\end{{{env}}}",
        "",
    ]
    return "\n".join(lines), notes, spreads


def default_caption(tune_sizes: set, pass_name: str, times: bool) -> str:
    if len(tune_sizes) == 1:
        n = next(iter(tune_sizes))
        exp = n.bit_length() - 1
        size = rf"$N = 2^{{{exp}}}$" if 2**exp == n else rf"$N = {n}$"
    else:
        size = "the tuning size"
    text = (rf"Parameters chosen by tuning for the dense {pass_name} pass at {size}, $d = 64$: base case size "
            r"$b$, serial grain $g$ and processor budget $p$, minimizing median runtime over the grid.")
    if times:
        text += r" Time is the chosen configuration's median runtime."
    return text + r" -- marks a parameter the algorithm does not take."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="benchmark.py --results-dir root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the .tex file into")
    parser.add_argument("--pass", dest="pass_name", choices=PASSES, default="forward",
                        help="Tabulate forward or backward (*_bwd) tuning (default: %(default)s)")
    parser.add_argument("--labels", default="skx,icx,spr",
                        help="Comma-separated machine labels, one column each (default: %(default)s)")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated algorithms, in row order (default: forward "
                             f"{DEFAULT_ALGORITHMS['forward']}; backward {DEFAULT_ALGORITHMS['backward']})")
    parser.add_argument("--times", action="store_true",
                        help="Add each chosen configuration's median runtime (widest; pair with --wide)")
    parser.add_argument("--wide", action="store_true", help="Use table* to span both columns")
    parser.add_argument("--name", default=None, help="Output file stem (default: dense_<pass>_tuning_table)")
    parser.add_argument("--caption", default=None, help="Override the generated caption")
    parser.add_argument("--latex-label", default=None, help=r"\label key (default: tab:dense-<pass>-tuning)")
    args = parser.parse_args()

    labels = [label for label in args.labels.split(",") if label]
    algorithms = [a for a in (args.algorithms or DEFAULT_ALGORITHMS[args.pass_name]).split(",") if a]
    table, notes, spreads = build_table(args.results_dir, labels, algorithms, args.pass_name, args.times,
                                        args.caption, args.latex_label or f"tab:dense-{args.pass_name}-tuning",
                                        args.wide)
    for note in notes:
        print(f"Note: {note}", file=sys.stderr)

    print(f"Tuning spread ({args.pass_name}): best and worst configuration per algorithm")
    for line in spreads:
        print("  " + line)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.name or f'dense_{args.pass_name}_tuning_table'}.tex"
    out.write_text(table)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
