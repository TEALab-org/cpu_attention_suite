#!/usr/bin/env python3
"""Generate a LaTeX table of runtime and cache misses at one sequence length.

Rows are grouped by architecture (one group per label), one row per algorithm; columns are
the median runtime from the timing sweep and the median L1/L2/L3 misses from the PAPI sweep.
--memory adds the peak resident memory each algorithm adds above its inputs (GiB), measured on
the first solve of the timing sweep's experiment.
Each cache-miss column gets a single power-of-ten scale in its header so values stay short,
and the best (lowest) value in each architecture group is set in bold. Missing results are
shown as "--". The table uses table* (spanning both columns of a two-column paper; pass
--single-column for table), with the caption below the tabular.
Requires \\usepackage{booktabs} and \\usepackage{multirow}.

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
from results_io import (  # noqa: E402
    ARCHITECTURES, PAPI_EVENTS, PASSES, display_name, load_records, machine_threads, result_kind,
)

DEFAULT_ALGORITHMS = {
    "forward": "fa2,summary,query,hybrid",
    "backward": "fa2_lockfree_bwd,fa2_locking_bwd,summary_bwd,owned_bwd,hybrid_bwd",
}

# (key, short header, unit); "time" and "memory" come from the timing sweep, the rest from the PAPI sweep
SWEEP_KEYS = {"time", "memory"}
MISS_COLUMNS = [(event, level) for event, level in PAPI_EVENTS]


def table_columns(memory: bool) -> List[Tuple[str, str, str]]:
    """Leading (non-miss) columns: runtime, then memory with --memory."""
    return [("time", "Runtime", "(s)")] + ([("memory", "Memory", "(GiB)")] if memory else [])


def find_value(records: List[dict], algorithm: str, n: int, key: str) -> Optional[Tuple[float, bool]]:
    """Median of the metric for (algorithm, N) and whether it comes from an unfinished experiment."""
    for rec in records:
        if rec.get("name") != algorithm or rec.get("N") != n:
            continue
        if key == "time":
            values = [t / 1000.0 for t in rec.get("times", [])]
        elif key == "memory":
            # One measurement per experiment; without a peak reset it is the process-wide peak instead
            memory = rec.get("memory") or {}
            values = [memory["peak_delta_bytes"] / 2**30] if memory.get("peak_reset") else []
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
                bold_best: bool, caption: Optional[str], label: str, wide: bool,
                pass_name: str, memory: bool = False) -> Tuple[str, List[str]]:
    lead = table_columns(memory)
    columns = [(key, header) for key, header, _ in lead] + MISS_COLUMNS
    warnings = []
    values: Dict[Tuple[str, str, str], Tuple[float, bool]] = {}
    for arch in labels:
        records = {kind: load_records(results_dir / arch / result_kind(kind, pass_name), include_partial)
                   for kind in ("sweep", "papi")}
        missing = {"timing": [], "memory": [], "PAPI": []}
        for algorithm in algorithms:
            for key, _ in columns:
                found = find_value(records["sweep" if key in SWEEP_KEYS else "papi"], algorithm, n, key)
                if found is None:
                    kind = {"time": "timing", "memory": "memory"}.get(key, "PAPI")
                    if algorithm not in missing[kind]:
                        missing[kind].append(algorithm)
                else:
                    values[(arch, algorithm, key)] = found
        for kind, algs in missing.items():
            if algs:
                warnings.append(f"{arch}: no {kind} results at N = {n} for {', '.join(algs)}")

    # Runtime and memory print in their units; cache-miss columns share one scale each, shown in the header
    exponents = {key: 0 if key in SWEEP_KEYS else column_exponent([v for (_, _, k), (v, _) in values.items() if k == key])
                 for key, _ in columns}
    n_lead, n_miss = len(lead), len(MISS_COLUMNS)

    any_partial = any(partial for _, partial in values.values())
    env = "table*" if wide else "table"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{ll" + "r" * len(columns) + "}",
        r"    \toprule",
        "    & & " + " & ".join(header for _, header, _ in lead) + rf" & \multicolumn{{{n_miss}}}{{c}}{{Cache misses}} \\",
        rf"    \cmidrule(lr){{{3 + n_lead}-{2 + n_lead + n_miss}}}",
        # Second header row: units for the leading columns, the power-of-ten scale for each miss column
        "    Architecture & Algorithm & " + " & ".join(
            [unit for _, _, unit in lead]
            + [f"{header} ($10^{{{exponents[key]}}}$)" if exponents[key] else header
               for key, header in MISS_COLUMNS]) + r" \\",
        r"    \midrule",
    ]
    for g, arch in enumerate(labels):
        best = {}
        for key, _ in columns:
            group = [values[(arch, a, key)][0] for a in algorithms if (arch, a, key) in values]
            best[key] = min(group) if bold_best and len(group) > 1 else None

        threads = machine_threads(results_dir, arch)
        name = ARCHITECTURES.get(arch, arch)
        arch_cell = (rf"\multirow{{{len(algorithms)}}}{{*}}{{\begin{{tabular}}[c]{{@{{}}l@{{}}}}{name}\\"
                     rf"{threads} cores\end{{tabular}}}}" if threads else rf"\multirow{{{len(algorithms)}}}{{*}}{{{name}}}")
        for i, algorithm in enumerate(algorithms):
            cells = [arch_cell if i == 0 else "", display_name(algorithm)]
            for key, _ in columns:
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
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption or default_caption(n, bold_best, any_partial, pass_name, memory)}}}",
        rf"  \label{{{label}}}",
        rf"\end{{{env}}}",
        "",
    ]
    return "\n".join(lines), warnings


def default_caption(n: int, bold_best: bool, any_partial: bool, pass_name: str, memory: bool = False) -> str:
    exp = n.bit_length() - 1
    size = rf"$N = 2^{{{exp}}}$" if 2**exp == n else rf"$N = {n}$"
    text = (rf"Dense {pass_name} pass at {size}, $d = 64$, using all cores of each node. Runtime is the median "
            r"over repetitions of the timing sweep; cache misses are medians from the PAPI sweep, "
            r"summed over all worker threads.")
    if memory:
        inputs = r"$Q$, $K$, $V$" if pass_name == "forward" else r"$Q$, $K$, $V$, $O$, $dO$"
        text += (rf" Memory is the peak resident memory the algorithm adds above its inputs ({inputs}) "
                 r"during its first solve, including its outputs.")
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
    parser.add_argument("--pass", dest="pass_name", choices=PASSES, default="forward",
                        help="Tabulate forward or backward (*_bwd) results (default: %(default)s)")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated algorithms, in row order (default: forward "
                             f"{DEFAULT_ALGORITHMS['forward']}; backward {DEFAULT_ALGORITHMS['backward']})")
    parser.add_argument("--name", default=None, help="Output file stem (default: dense_<pass>_table_N<n>)")
    parser.add_argument("--caption", default=None, help="Override the generated caption")
    parser.add_argument("--latex-label", default=None, help=r"\label key (default: tab:dense-<pass>-N<n>, plus -memory with --memory)")
    parser.add_argument("--include-partial", action="store_true",
                        help="Use unfinished experiments too, marked with a dagger")
    parser.add_argument("--single-column", action="store_true",
                        help="Use table instead of the default table* (which spans both columns)")
    parser.add_argument("--no-bold", action="store_true", help="Do not bold the best value per architecture")
    parser.add_argument("--memory", action="store_true",
                        help="Add a column with each algorithm's peak memory above its inputs (GiB); "
                             "default stem gets a _memory suffix")
    args = parser.parse_args()

    labels = [label for label in args.labels.split(",") if label]
    algorithms = [a for a in (args.algorithms or DEFAULT_ALGORITHMS[args.pass_name]).split(",") if a]
    table, warnings = build_table(args.results_dir, labels, algorithms, args.n, args.include_partial,
                                  not args.no_bold, args.caption,
                                  args.latex_label or f"tab:dense-{args.pass_name}-N{args.n}" + ("-memory" if args.memory else ""),
                                  not args.single_column, args.pass_name, args.memory)
    for warning in warnings:
        print(f"Note: {warning}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dense_{args.pass_name}_table_N{args.n}" + ("_memory" if args.memory else "")
    out = args.output_dir / f"{args.name or stem}.tex"
    out.write_text(table)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
