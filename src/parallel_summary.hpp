#pragma once

#include "attention_utilities.hpp"
#include <cilk/cilk.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

/**
 * Contiguous 0-based index interval [start, start + len)
 */
struct IndexRange {
    Eigen::Index start;
    Eigen::Index len;
};

/**
 * Summary structure \Sigma wrapping (m, \ell, U) over a row interval I.
 *
 * Sizing \Sigma to |I| rather than full N bounds the total auxiliary memory
 * across all active recursive branches to O(\sqrt{p} * N * d) instead of O(p * N * d).
 */
struct AttentionSummary {
    Eigen::Index row_offset = 0;
    Eigen::VectorXf m;
    Eigen::VectorXf l;
    AttentionMatrixf U;

    AttentionSummary() = default;

    /// Allocates summary buffers covering the interval I = [start, start + len)
    explicit AttentionSummary(IndexRange I)
        : row_offset(I.start),
          m(I.len),
          l(I.len),
          U(I.len, 64) {}

    // Global index accessors (i \in I) matching pseudocode notation \Sigma.m_i, \Sigma.\ell_i, \Sigma.U_i
    [[nodiscard]] float& m_at(Eigen::Index i) noexcept { return m(i - row_offset); }
    [[nodiscard]] float m_at(Eigen::Index i) const noexcept { return m(i - row_offset); }

    [[nodiscard]] float& l_at(Eigen::Index i) noexcept { return l(i - row_offset); }
    [[nodiscard]] float l_at(Eigen::Index i) const noexcept { return l(i - row_offset); }

    [[nodiscard]] auto U_row(Eigen::Index i) noexcept { return U.row(i - row_offset); }
    [[nodiscard]] auto U_row(Eigen::Index i) const noexcept { return U.row(i - row_offset); }
};

namespace detail {

/**
 * Preallocated thread-local scratch space for serial base-case leaf evaluations.
 */
struct LeafWorkspace {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S;
    AttentionMatrixf u_hat;

    explicit LeafWorkspace(Eigen::Index b) : S(b, b), u_hat(b, 64) {}
};

/**
 * Serial recursive cache-oblivious accumulator: DenseAccumulate(I, J, \Sigma)
 * Invoked once the parallel recursion reaches p == 1.
 */
inline void dense_accumulate(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    IndexRange I,
    IndexRange J,
    AttentionSummary& sigma,
    Eigen::Index b,
    float scale,
    LeafWorkspace& ws
) {
    if (I.len <= 0 || J.len <= 0) {
        return;
    }

    // Base Case: |I| <= b and |J| <= b
    if (I.len <= b && J.len <= b) {
        auto Q_I = Q.middleRows(I.start, I.len);
        auto K_J = K.middleRows(J.start, J.len);
        auto V_J = V.middleRows(J.start, J.len);

        // S_{I,J} <- (Q_I * K_J^T) / sqrt(d)
        auto S_block = ws.S.topLeftCorner(I.len, J.len);
        S_block.noalias() = (Q_I * K_J.transpose()) * scale;

        Eigen::VectorXf m_hat(I.len);
        Eigen::VectorXf l_hat(I.len);

        for (Eigen::Index r = 0; r < I.len; ++r) {
            auto s_row = S_block.row(r);
            const float m_hat_i = s_row.maxCoeff();
            m_hat(r) = m_hat_i;

            s_row = (s_row.array() - m_hat_i).exp();
            l_hat(r) = s_row.sum();
        }

        // \hat{U}_I <- \exp(S_{I,J} - \hat{m}_I) * V_J
        auto u_hat_block = ws.u_hat.topRows(I.len);
        u_hat_block.noalias() = S_block * V_J;

        // Associative update: \Sigma_i <- \Sigma_i \bigoplus (\hat{m}_i, \hat{\ell}_i, \hat{u}_i)
        for (Eigen::Index r = 0; r < I.len; ++r) {
            const Eigen::Index i = I.start + r;
            const float m_L = sigma.m_at(i);
            const float l_L = sigma.l_at(i);
            const float m_R = m_hat(r);
            const float l_R = l_hat(r);

            float m_new = 0.0f;
            float alpha = 0.0f;
            float beta = 0.0f;

            if (l_L == 0.0f) {
                m_new = m_R;
                alpha = 0.0f;
                beta = 1.0f;
            } else if (l_R == 0.0f) {
                m_new = m_L;
                alpha = 1.0f;
                beta = 0.0f;
            } else {
                m_new = std::max(m_L, m_R);
                alpha = std::exp(m_L - m_new);
                beta = std::exp(m_R - m_new);
            }

            sigma.m_at(i) = m_new;
            sigma.l_at(i) = alpha * l_L + beta * l_R;
            sigma.U_row(i).noalias() = alpha * sigma.U_row(i) + beta * u_hat_block.row(r);
        }
        return;
    }

    // Split I = I_1 \cup I_2, J = J_1 \cup J_2 into consecutive halves
    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    const IndexRange I1{I.start, i_mid};
    const IndexRange I2{I.start + i_mid, I.len - i_mid};
    const IndexRange J1{J.start, j_mid};
    const IndexRange J2{J.start + j_mid, J.len - j_mid};

    dense_accumulate(Q, K, V, I1, J1, sigma, b, scale, ws);
    dense_accumulate(Q, K, V, I1, J2, sigma, b, scale, ws);
    dense_accumulate(Q, K, V, I2, J1, sigma, b, scale, ws);
    dense_accumulate(Q, K, V, I2, J2, sigma, b, scale, ws);
}

/**
 * ParallelForwardMerge(I, J, \Sigma, \Sigma_{11}, \Sigma_{12}, \Sigma_{21}, \Sigma_{22})
 * Merges four quadrant child summaries into the parent summary \Sigma.
 */
inline void parallel_forward_merge(
    IndexRange I,
    IndexRange /*J*/,
    AttentionSummary& sigma,
    const AttentionSummary& sigma_11,
    const AttentionSummary& sigma_12,
    const AttentionSummary& sigma_21,
    const AttentionSummary& sigma_22
) {
    const Eigen::Index i_mid = I.len / 2;
    const IndexRange I1{I.start, i_mid};

    // Flattened parallel loop over a \in {1, 2} and i \in I_a
    cilk_for (Eigen::Index r = 0; r < I.len; ++r) {
        const Eigen::Index i = I.start + r;
        const bool in_first_half = (r < I1.len);

        // Select (\Sigma_{a1}, \Sigma_{a2}) based on whether i \in I_1 (a=1) or i \in I_2 (a=2)
        const AttentionSummary& sigma_a1 = in_first_half ? sigma_11 : sigma_21;
        const AttentionSummary& sigma_a2 = in_first_half ? sigma_12 : sigma_22;

        const float m_L = sigma_a1.m_at(i);
        const float m_R = sigma_a2.m_at(i);
        const float l_L = sigma_a1.l_at(i);
        const float l_R = sigma_a2.l_at(i);

        float m = 0.0f;
        float alpha = 0.0f;
        float beta = 0.0f;

        if (l_L == 0.0f) {
            m = m_R;
            alpha = 0.0f;
            beta = 1.0f;
        } else if (l_R == 0.0f) {
            m = m_L;
            alpha = 1.0f;
            beta = 0.0f;
        } else {
            m = std::max(m_L, m_R);
            alpha = std::exp(m_L - m);
            beta = std::exp(m_R - m);
        }

        sigma.m_at(i) = m;
        sigma.l_at(i) = alpha * l_L + beta * l_R;

        // Vectorized execution of: for k = 1,...,d: \Sigma.U_{i,k} <- \alpha \Sigma_{a1}.U_{i,k} + \beta \Sigma_{a2}.U_{i,k}
        sigma.U_row(i).noalias() = alpha * sigma_a1.U_row(i) + beta * sigma_a2.U_row(i);
    }
}

/**
 * ParallelForwardSummary(I, J, p, \Sigma)
 * Recursively spawns 4 quadrant subtasks until processor budget p == 1,
 * then executes serial cache-oblivious accumulation and merges upward in parallel.
 */
inline void parallel_forward_summary(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    IndexRange I,
    IndexRange J,
    int p,
    AttentionSummary& sigma,
    Eigen::Index b,
    float scale
) {
    if (I.len <= 0 || J.len <= 0) {
        return;
    }

    // Base Case: p == 1 (or subproblem has already reached leaf granularity)
    if (p <= 1 || (I.len <= b && J.len <= b)) {
        // parallel for i \in I: (\Sigma.m_i, \Sigma.\ell_i) <- (-\infty, 0)
        cilk_for (Eigen::Index r = 0; r < I.len; ++r) {
            const Eigen::Index i = I.start + r;
            sigma.m_at(i) = -std::numeric_limits<float>::infinity();
            sigma.l_at(i) = 0.0f;
        }

        // parallel for (i, k) \in I \times |d|: \Sigma.U_{i, k} <- 0
        cilk_for (Eigen::Index r = 0; r < I.len; ++r) {
            const Eigen::Index i = I.start + r;
            sigma.U_row(i).setZero();
        }

        // Thread-local workspace for the serial recursive traversal
        LeafWorkspace ws(b);
        dense_accumulate(Q, K, V, I, J, sigma, b, scale, ws);
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
    // Automatically freed via RAII when exiting scope after parallel_forward_merge
    AttentionSummary sigma_11(I1);
    AttentionSummary sigma_12(I1);
    AttentionSummary sigma_21(I2);
    AttentionSummary sigma_22(I2);

    const int child_p = p / 4;

    // 4-way parallel divide-and-conquer (spawn 3 children, run 4th on current worker)
    cilk_spawn parallel_forward_summary(Q, K, V, I1, J1, child_p, sigma_11, b, scale);
    cilk_spawn parallel_forward_summary(Q, K, V, I1, J2, child_p, sigma_12, b, scale);
    cilk_spawn parallel_forward_summary(Q, K, V, I2, J1, child_p, sigma_21, b, scale);
               parallel_forward_summary(Q, K, V, I2, J2, child_p, sigma_22, b, scale);
    cilk_sync;

    // Merge the 4 quadrant summaries into \Sigma
    parallel_forward_merge(I, J, sigma, sigma_11, sigma_12, sigma_21, sigma_22);
}

} // namespace detail

/**
 * Top-Level OpenCilk Cache-Oblivious Attention Driver:
 *   1. Runs ParallelForwardSummary([0, N), [0, N), p, \Sigma)
 *   2. Normalizes O_{i,:} <- \Sigma.U_i / \Sigma.\ell_i in parallel
 *
 * @param p Processor subproblem budget (assumed to be a power of 4: 1, 4, 16, 64, ...)
 * @param b Base case block threshold for DenseAccumulate
 */
inline AttentionMatrixf cache_oblivious_attention_opencilk(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    int p = 16,
    Eigen::Index b = 64
) {
    // Verify p is a power of 4 (p >= 1, power of 2, and single bit is in an even position)
    assert(p >= 1 && (p & (p - 1)) == 0 && (p & 0x55555555) != 0 && "p must be a power of 4");

    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));
    const IndexRange full_range{0, N};

    // Allocate root summary \Sigma covering [0, N)
    AttentionSummary sigma(full_range);

    // 1. Compute unnormalized summary (m, \ell, U) via 4-way recursive task tree
    detail::parallel_forward_summary(Q, K, V, full_range, full_range, p, sigma, b, scale);

    // 2. Final parallel normalization: O_{i,:} <- \Sigma.U_i / \Sigma.\ell_i
    AttentionMatrixf O(N, 64);
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        const float l_i = sigma.l_at(i);
        if (l_i > 0.0f) {
            O.row(i).noalias() = sigma.U_row(i) / l_i;
        } else {
            O.row(i).setZero();
        }
    }

    return O;
}
