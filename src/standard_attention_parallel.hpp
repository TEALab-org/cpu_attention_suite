#pragma once

#include "attention_utilities.hpp"
#include <cmath>
#include <limits>
#include <cilk/cilk.h>

/**
 * OpenCilk-Parallelized Standard Attention Reference:
 *   S = (Q * K^T) / sqrt(d)
 *   P = softmax(S, row-wise)
 *   O = P * V
 *
 * Parallelized over tokens across three distinct phases. Materializes
 * S in global memory to benchmark the quadratic memory footprint O(N^2).
 */
inline AttentionMatrixf standard_attention_parallel(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    bool causal = false
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));

    // 1. Materialize the full N x N intermediate matrix in RowMajor order
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S(N, N);

    // Phase 1: Parallel Affinity Scores S_i = (q_i * K^T) * scale
    // Each worker thread computes an independent (1 x 64) * (64 x N) vector-matrix product.
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        S.row(i).noalias() = (Q.row(i) * K.transpose()) * scale;
    }

    // Phase 2: Parallel Numerically Stable Row-wise Softmax
    // Completely independent per row; no write hazards or synchronizations.
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        auto row = S.row(i);

        if (causal && i + 1 < N) {
            row.tail(N - 1 - i).setConstant(-std::numeric_limits<float>::infinity());
        }

        const float m_i = row.maxCoeff();
        row = (row.array() - m_i).exp();
        const float l_i = row.sum();

        if (l_i > 0.0f) {
            row.array() /= l_i;
        } else {
            row.setZero();
        }
    }

    // Phase 3: Parallel Output Projection O_i = P_i * V
    // Each worker computes an independent (1 x N) * (N x 64) matrix product.
    AttentionMatrixf O(N, 64);
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        O.row(i).noalias() = S.row(i) * V;
    }

    return O;
}
