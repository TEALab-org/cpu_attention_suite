#pragma once

#include "dense_accumulate.hpp"
#include <cmath>
#include <utility>

/**
 * Return tuple (O, m, \ell) matching DenseAttention pseudocode.
 */
struct DenseAttentionResult {
    AttentionMatrixf O;
    Eigen::VectorXf m;
    Eigen::VectorXf l;
};

/**
 * Serial Cache-Oblivious Recursive Attention: DenseAttention(Q, K, V)
 * Returns the full (O, m, \ell) state tuple.
 */
inline DenseAttentionResult dense_attention_recursive(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    Eigen::Index b = 64
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));
    const IndexRange full_range{0, N};

    // 1. Initialize (m_i, \ell_i, u_i) <- (-\infty, 0, 0) for i = 0, ..., N - 1
    AttentionSummary sigma(full_range);
    sigma.reset();

    // Preallocate leaf workspace once (sized to base case threshold b)
    detail::LeafWorkspace ws(b);

    // 2. Recursive divide-and-conquer accumulation over [0, N) x [0, N)
    detail::dense_accumulate(Q, K, V, full_range, full_range, sigma, b, scale, ws);

    // 3. Final row-wise normalization: O_{i,:} <- u_i / \ell_i
    AttentionMatrixf O(N, 64);
    for (Eigen::Index i = 0; i < N; ++i) {
        if (sigma.l(i) > 0.0f) {
            O.row(i).noalias() = sigma.U.row(i) / sigma.l(i);
        } else {
            O.row(i).setZero();
        }
    }

    return DenseAttentionResult{std::move(O), std::move(sigma.m), std::move(sigma.l)};
}
