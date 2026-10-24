#pragma once

#include "attention_utilities.hpp"
#include <algorithm>
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

    /// (m_i, \ell_i, U_i) <- (-\infty, 0, 0) for all i \in I
    void reset() {
        m.setConstant(-std::numeric_limits<float>::infinity());
        l.setZero();
        U.setZero();
    }

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
 * Preallocated scratch space for serial base-case leaf evaluations.
 * Avoids O((N / b)^2) dynamic heap allocations during recursion.
 */
struct LeafWorkspace {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S;
    AttentionMatrixf u_hat;
    Eigen::VectorXf m_hat;
    Eigen::VectorXf l_hat;

    explicit LeafWorkspace(Eigen::Index b) : S(b, b), u_hat(b, 64), m_hat(b), l_hat(b) {}
};

/**
 * Associative softmax reduction operator \bigoplus:
 *   (m, \ell, u) <- (m_L, \ell_L, u_L) \bigoplus (m_R, \ell_R, u_R)
 *
 * An operand with \ell == 0 is the identity (-\infty, 0, 0) and is skipped, which
 * also avoids evaluating exp(-\infty - -\infty) = NaN when both sides are empty.
 * u may alias u_L (in-place accumulation) but not u_R.
 */
template <typename OutRow, typename LeftRow, typename RightRow>
inline void combine_softmax_state(
    float& m,
    float& l,
    OutRow&& u,
    float m_L,
    float l_L,
    const LeftRow& u_L,
    float m_R,
    float l_R,
    const RightRow& u_R
) {
    if (l_L == 0.0f) {
        m = m_R;
        l = l_R;
        u = u_R;
    } else if (l_R == 0.0f) {
        m = m_L;
        l = l_L;
        u = u_L;
    } else {
        const float m_new = std::max(m_L, m_R);
        const float alpha = std::exp(m_L - m_new);
        const float beta = std::exp(m_R - m_new);
        m = m_new;
        l = alpha * l_L + beta * l_R;
        u = alpha * u_L + beta * u_R;
    }
}

/**
 * Serial recursive cache-oblivious accumulator: DenseAccumulate(I, J, \Sigma)
 *
 * Folds the contribution of keys/values J into the summary of queries I.
 * \Sigma must cover I and be initialized (see AttentionSummary::reset).
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

        // Compute \hat{m}_i and \hat{\ell}_i, transforming S_block in-place into exp(s_{ij} - \hat{m}_i)
        for (Eigen::Index r = 0; r < I.len; ++r) {
            auto s_row = S_block.row(r);
            const float m_hat_i = s_row.maxCoeff();
            ws.m_hat(r) = m_hat_i;

            s_row = (s_row.array() - m_hat_i).exp();
            ws.l_hat(r) = s_row.sum();
        }

        // \hat{U}_I <- \exp(S_{I,J} - \hat{m}_I) * V_J, computed for all i \in I via one GEMM
        auto u_hat_block = ws.u_hat.topRows(I.len);
        u_hat_block.noalias() = S_block * V_J;

        // Associative update: \Sigma_i <- \Sigma_i \bigoplus (\hat{m}_i, \hat{\ell}_i, \hat{u}_i)
        for (Eigen::Index r = 0; r < I.len; ++r) {
            const Eigen::Index i = I.start + r;
            combine_softmax_state(
                sigma.m_at(i), sigma.l_at(i), sigma.U_row(i),
                sigma.m_at(i), sigma.l_at(i), sigma.U_row(i),
                ws.m_hat(r), ws.l_hat(r), u_hat_block.row(r)
            );
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

} // namespace detail
