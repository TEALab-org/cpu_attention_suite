#!/usr/bin/env python3
"""Generate synthetic stacked (3, N, d) attention inputs [Q, K, V] in .npy format."""
# Writen by Gemini (Oct 6 2026)

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def generate_problem(n: int, d: int, seed: int | None = None) -> np.ndarray:
    """Return a (3, n, d) float32 tensor [Q, K, V] uniformly distributed in [-1.0, 1.0)."""
    if n <= 0 or d <= 0:
        raise ValueError("Sequence length N and head dimension d must be positive integers.")
    rng = np.random.default_rng(seed)
    # Axis 0 contains [Q, K, V] sequentially
    return rng.uniform(-1.0, 1.0, size=(3, n, d)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic QKV attention problem saved as a (3, N, d) float32 .npy file."
    )
    parser.add_argument(
        "-N",
        "--seq-len",
        dest="n",
        type=int,
        required=True,
        help="Sequence length (number of tokens / rows per matrix).",
    )
    parser.add_argument(
        "-d",
        "--head-dim",
        dest="d",
        type=int,
        required=True,
        help="Head dimension (number of columns per matrix, e.g., 64).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=True,
        help="Destination path for the .npy file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible benchmark sets.",
    )

    args = parser.parse_args()

    tensor_qkv = generate_problem(args.n, args.d, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, tensor_qkv)

    # Quick diagnostic report
    file_bytes = output_path.stat().st_size
    print(f"Wrote attention problem to: {output_path.resolve()}")
    print(f"  Shape:       {tensor_qkv.shape} (Axis 0: Q=0, K=1, V=2)")
    print(f"  Dtype:       {tensor_qkv.dtype} (C-contiguous / row-major)")
    print(f"  Buffer size: {tensor_qkv.nbytes / (1024 * 1024):.2f} MB ({file_bytes} bytes on disk)")


if __name__ == "__main__":
    main()
