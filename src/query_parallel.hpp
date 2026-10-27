#pragma once

#include "dense_accumulate.hpp"
#include <cilk/cilk.h>
#include <cmath>

namespace detail {

/**
 * QueryParallelAccumulate(I, J, \Sigma)
 *
 * 2-way parallel recursion over query halves sharing a single summary \Sigma: each phase
 * spawns the two I halves against the same J half, so concurrent tasks write disjoint rows.
 * Once |I| <= grain the remaining (I, J) block runs as a serial DenseAccumulate.
 * \Sigma must be initialized before the first call; the recursion never resets it.
 */
inline void query_parallel_accumulate(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    IndexRange I,
    IndexRange J,
    AttentionSummary& sigma,
    Eigen::Index b,
    Eigen::Index grain,
    float scale,
    LeafWorkspaces& workspaces
) {
    if (I.len <= 0 || J.len <= 0) {
        return;
    }

    // Base Case: |I| <= grain, serial cache-oblivious traversal with this worker's workspace
    if (I.len <= grain) {
        LeafWorkspace& ws = workspaces;
        ws.reserve(b);
        dense_accumulate(Q, K, V, I, J, sigma, b, scale, ws);
        return;
    }

    // Split I = I_1 \cup I_2, J = J_1 \cup J_2 into consecutive halves
    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    const IndexRange I1{I.start, i_mid};
    const IndexRange I2{I.start + i_mid, I.len - i_mid};
    const IndexRange J1{J.start, j_mid};
    const IndexRange J2{J.start + j_mid, J.len - j_mid};

    cilk_spawn query_parallel_accumulate(Q, K, V, I1, J1, sigma, b, grain, scale, workspaces);
               query_parallel_accumulate(Q, K, V, I2, J1, sigma, b, grain, scale, workspaces);
    cilk_sync;

    cilk_spawn query_parallel_accumulate(Q, K, V, I1, J2, sigma, b, grain, scale, workspaces);
               query_parallel_accumulate(Q, K, V, I2, J2, sigma, b, grain, scale, workspaces);
    cilk_sync;
}

} // namespace detail

/**
 * Query-Parallel Cache-Oblivious Attention Driver:
 *   1. Initializes \Sigma over [0, N) to (-\infty, 0, 0) in parallel
 *   2. Runs QueryParallelAccumulate([0, N), [0, N), \Sigma)
 *   3. Normalizes O_{i,:} <- \Sigma.U_i / \Sigma.\ell_i in parallel
 *
 * Unlike ParallelSummary this needs no processor budget and no auxiliary summaries.
 *
 * @param b     Base case block threshold for DenseAccumulate
 * @param grain Query-range size below which recursion continues serially (>= b recommended)
 */
inline AttentionMatrixf query_parallel_attention(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    Eigen::Index b = 64,
    Eigen::Index grain = 256
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));
    const IndexRange full_range{0, N};

    AttentionSummary sigma(full_range);
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        sigma.m(i) = -std::numeric_limits<float>::infinity();
        sigma.l(i) = 0.0f;
        sigma.U.row(i).setZero();
    }

    detail::LeafWorkspaces workspaces;
    detail::query_parallel_accumulate(Q, K, V, full_range, full_range, sigma, b, grain, scale, workspaces);

    AttentionMatrixf O(N, 64);
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        const float l_i = sigma.l(i);
        if (l_i > 0.0f) {
            O.row(i).noalias() = sigma.U.row(i) / l_i;
        } else {
            O.row(i).setZero();
        }
    }

    return O;
}
