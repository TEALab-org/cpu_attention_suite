#pragma once

#include "dense_accumulate.hpp"
#include "parallel_summary.hpp"
#include "query_parallel.hpp"
#include <cilk/cilk.h>
#include <cmath>
#include <limits>

namespace detail {

/**
 * HybridForwardSummary(I, J, p, \Sigma)
 *
 * Follows the ParallelSummary 4-way recursion while the processor budget p divides evenly
 * by 4, then hands each remaining subproblem to QueryParallelAccumulate with a fresh \Sigma.
 * The number of summary levels is the number of times 4 divides p (e.g. 48, 80, 112 -> 2),
 * and work stealing balances the 4^k query-parallel subproblems, so p need not be a power of 4.
 */
inline void hybrid_forward_summary(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    IndexRange I,
    IndexRange J,
    int p,
    AttentionSummary& sigma,
    Eigen::Index b,
    Eigen::Index grain,
    float scale
) {
    if (I.len <= 0 || J.len <= 0) {
        return;
    }

    // Switch: p no longer splits evenly 4 ways (or subproblem has already reached leaf granularity)
    if (p % 4 != 0 || (I.len <= b && J.len <= b)) {
        // parallel for i \in I: (\Sigma.m_i, \Sigma.\ell_i, \Sigma.U_i) <- (-\infty, 0, 0)
        cilk_for (Eigen::Index r = 0; r < I.len; ++r) {
            const Eigen::Index i = I.start + r;
            sigma.m_at(i) = -std::numeric_limits<float>::infinity();
            sigma.l_at(i) = 0.0f;
            sigma.U_row(i).setZero();
        }

        query_parallel_accumulate(Q, K, V, I, J, sigma, b, grain, scale);
        return;
    }

    // Split I = I_1 \cup I_2 and J = J_1 \cup J_2 into consecutive halves
    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    const IndexRange I1{I.start, i_mid};
    const IndexRange I2{I.start + i_mid, I.len - i_mid};
    const IndexRange J1{J.start, j_mid};
    const IndexRange J2{J.start + j_mid, J.len - j_mid};

    // Allocate \Sigma_{11}, \Sigma_{12}, \Sigma_{21}, \Sigma_{22}
    AttentionSummary sigma_11(I1);
    AttentionSummary sigma_12(I1);
    AttentionSummary sigma_21(I2);
    AttentionSummary sigma_22(I2);

    const int child_p = p / 4;

    cilk_spawn hybrid_forward_summary(Q, K, V, I1, J1, child_p, sigma_11, b, grain, scale);
    cilk_spawn hybrid_forward_summary(Q, K, V, I1, J2, child_p, sigma_12, b, grain, scale);
    cilk_spawn hybrid_forward_summary(Q, K, V, I2, J1, child_p, sigma_21, b, grain, scale);
               hybrid_forward_summary(Q, K, V, I2, J2, child_p, sigma_22, b, grain, scale);
    cilk_sync;

    parallel_forward_merge(I, J, sigma, sigma_11, sigma_12, sigma_21, sigma_22);
}

} // namespace detail

/**
 * Hybrid ParallelSummary / QueryParallel Attention Driver:
 *   1. Runs HybridForwardSummary([0, N), [0, N), p, \Sigma)
 *   2. Normalizes O_{i,:} <- \Sigma.U_i / \Sigma.\ell_i in parallel
 *
 * @param p     Processor budget (any p >= 1)
 * @param b     Base case block threshold for DenseAccumulate
 * @param grain Query-range size below which QueryParallel recursion continues serially
 */
inline AttentionMatrixf hybrid_attention(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    int p,
    Eigen::Index b = 64,
    Eigen::Index grain = 256
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));
    const IndexRange full_range{0, N};

    AttentionSummary sigma(full_range);
    detail::hybrid_forward_summary(Q, K, V, full_range, full_range, p, sigma, b, grain, scale);

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
