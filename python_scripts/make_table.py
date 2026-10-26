#!/usr/bin/env python3
"""Generate a LaTeX table of runtime and cache misses at one sequence length.

Rows are grouped by architecture (one group per label), one row per algorithm; columns are
the median runtime from the timing sweep and the median L1/L2/L3 misses from the PAPI sweep.
Each cache-miss column gets a single power-of-ten scale in its header so values stay short,
and the best (lowest) value in each architecture group is set in bold. Missing results are
shown as "--". Requires \\usepackage{booktabs} and \\usepackage{multirow}.

Example:
    python python_scripts/make_table.py \\
        --results-dir ../cpu_attention_results --output-dir ../cpu_attention_output
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_io import ALGORITHMS, ARCHITECTURES, PAPI_EVENTS, load_records, machine_threads  # noqa: E402

DISPLAY = dict(ALGORITHMS)
DEFAULT_ALGORITHMS = "fa2,summary,query,hybrid"

# (key, short header); "time" comes from the timing sweep, the rest from the PAPI sweep
COLUMNS = [("time", "Runtime")] + [(event, level) for event, level in PAPI_EVENTS]


def find_value(records: List[dict], algorithm: str, n: int, key: str) -> Optional[Tuple[float, bool]]:
    """Median of the metric for (algorithm, N) and whether it comes from an unfinished experiment."""
    for rec in records:
        if rec.get("name") != algorithm or rec.get("N") != n:
            continue
        if key == "time":
            values = [t / 1000.0 for t in rec.get("times", [])]
        else:
            values = [float(c[key]) for c in rec.get("counters", []) if key in c]
        if values:
            return statistics.median(values), rec.get("status") != "complete"
    return None


def column_exponent(values: List[float]) -> int:
    """Power of ten (a multiple of 3) that keeps the smallest value in the column >= 1."""
    positive = [v for v in values if v > 0]
    if not positive:
        return 0
    return 3 * math.floor(math.log10(min(positive)) / 3)


def format_number(value: float, sig: int = 3) -> str:
    if value == 0:
        return "0"
    decimals = max(0, sig - 1 - math.floor(math.log10(abs(value))))
    return f"{value:,.{decimals}f}".replace(",", "{,}")


def build_table(results_dir: Path, labels: List[str], algorithms: List[str], n: int, include_partial: bool,
                bold_best: bool, caption: Optional[str], label: str, wide: bool) -> Tuple[str, List[str]]:
    warnings = []
    values: Dict[Tuple[str, str, str], Tuple[float, bool]] = {}
    for arch in labels:
        records = {kind: load_records(results_dir / arch / kind, include_partial) for kind in ("sweep", "papi")}
        missing = {"timing": [], "PAPI": []}
        for algorithm in algorithms:
            for key, _ in COLUMNS:
                found = find_value(records["sweep" if key == "time" else "papi"], algorithm, n, key)
                if found is None:
                    kind = "timing" if key == "time" else "PAPI"
                    if algorithm not in missing[kind]:
                        missing[kind].append(algorithm)
                else:
                    values[(arch, algorithm, key)] = found
        for kind, algs in missing.items():
            if algs:
                warnings.append(f"{arch}: no {kind} results at N = {n} for {', '.join(algs)}")

    # Runtimes print in seconds; cache-miss columns share one scale each, shown in the header
    exponents = {key: 0 if key == "time" else column_exponent([v for (_, _, k), (v, _) in values.items() if k == key])
                 for key, _ in COLUMNS}
    # Second header row: units for the runtime, the power-of-ten scale for each miss column
    units = ["(s)"] + [rf"($10^{{{exponents[key]}}}$)" if exponents[key] else "" for key, _ in COLUMNS[1:]]
    n_miss = len(COLUMNS) - 1

    any_partial = any(partial for _, partial in values.values())
    env = "table*" if wide else "table"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"  \centering",
        rf"  \caption{{{caption or default_caption(n, bold_best, any_partial)}}}",
        rf"  \label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{ll" + "r" * len(COLUMNS) + "}",
        r"    \toprule",
        rf"    & & Runtime & \multicolumn{{{n_miss}}}{{c}}{{Cache misses}} \\",
        rf"    \cmidrule(lr){{4-{3 + n_miss}}}",
        "    Architecture & Algorithm & " + " & ".join(
            [units[0]] + [f"{header} {unit}".strip() for (_, header), unit in zip(COLUMNS[1:], units[1:])]) + r" \\",
        r"    \midrule",
    ]
    for g, arch in enumerate(labels):
        best = {}
        for key, _ in COLUMNS:
            group = [values[(arch, a, key)][0] for a in algorithms if (arch, a, key) in values]
            best[key] = min(group) if bold_best and len(group) > 1 else None

        threads = machine_threads(results_dir, arch)
        name = ARCHITECTURES.get(arch, arch)
        arch_cell = (rf"\multirow{{{len(algorithms)}}}{{*}}{{\begin{{tabular}}[c]{{@{{}}l@{{}}}}{name}\\"
                     rf"{threads} cores\end{{tabular}}}}" if threads else rf"\multirow{{{len(algorithms)}}}{{*}}{{{name}}}")
        for i, algorithm in enumerate(algorithms):
            cells = [arch_cell if i == 0 else "", DISPLAY.get(algorithm, algorithm)]
            for key, _ in COLUMNS:
                if (arch, algorithm, key) not in values:
                    cells.append("--")
                    continue
                value, partial = values[(arch, algorithm, key)]
                text = format_number(value / 10 ** exponents[key])
                if best[key] is not None and value == best[key]:
                    text = rf"\textbf{{{text}}}"
                if partial:
                    text += r"$^\dagger$"
                cells.append(text)
            lines.append("    " + " & ".join(cells) + r" \\")
        if g + 1 < len(labels):
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}", rf"\end{{{env}}}", ""]
    return "\n".join(lines), warnings


def default_caption(n: int, bold_best: bool, any_partial: bool) -> str:
    exp = n.bit_length() - 1
    size = rf"$N = 2^{{{exp}}}$" if 2**exp == n else rf"$N = {n}$"
    text = (rf"Dense forward pass at {size}, $d = 64$, using all cores of each node. Runtime is the median "
            r"over repetitions of the timing sweep; cache misses are medians from the PAPI sweep, "
            r"summed over all worker threads.")
    if bold_best:
        text += r" The lowest value per architecture is in bold."
    text += r" -- marks results not yet available."
    if any_partial:
        text += r" $^\dagger$Not all repetitions finished."
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="benchmark.py --results-dir root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the .tex file into")
    parser.add_argument("--n", type=int, default=2**20, help="Sequence length to tabulate (default: %(default)s)")
    parser.add_argument("--labels", default="skx,icx,spr",
                        help="Comma-separated machine labels, one row group each (default: %(default)s)")
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS,
                        help="Comma-separated algorithms, in row order (default: %(default)s)")
    parser.add_argument("--name", default=None, help="Output file stem (default: dense_forward_table_N<n>)")
    parser.add_argument("--caption", default=None, help="Override the generated caption")
    parser.add_argument("--latex-label", default=None, help=r"\label key (default: tab:dense-forward-N<n>)")
    parser.add_argument("--include-partial", action="store_true",
                        help="Use unfinished experiments too, marked with a dagger")
    parser.add_argument("--wide", action="store_true",
                        help="Use table* to span both columns of a two-column layout")
    parser.add_argument("--no-bold", action="store_true", help="Do not bold the best value per architecture")
    args = parser.parse_args()

    labels = [label for label in args.labels.split(",") if label]
    algorithms = [a for a in args.algorithms.split(",") if a]
    table, warnings = build_table(args.results_dir, labels, algorithms, args.n, args.include_partial,
                                  not args.no_bold, args.caption, args.latex_label or f"tab:dense-forward-N{args.n}",
                                  args.wide)
    for warning in warnings:
        print(f"Note: {warning}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.name or f'dense_forward_table_N{args.n}'}.tex"
    out.write_text(table)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
