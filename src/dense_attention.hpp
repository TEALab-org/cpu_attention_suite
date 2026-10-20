#pragma once

#include "attention_utilities.hpp" // Provides AttentionMatrixf and AttentionProblem
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
 * Return tuple (O, m, \ell) matching DenseAttention pseudocode.
 */
struct DenseAttentionResult {
    AttentionMatrixf O;
    Eigen::VectorXf m;
    Eigen::VectorXf l;
};

namespace detail {

/**
 * Preallocated scratch space for base-case leaf evaluations.
 * Avoids O((N / b)^2) dynamic heap allocations during recursion.
 */
struct LeafWorkspace {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S;
    AttentionMatrixf u_hat;

    explicit LeafWorkspace(Eigen::Index b) : S(b, b), u_hat(b, 64) {}
};

/**
 * Associative softmax reduction operator \bigoplus:
 *   (m_i, \ell_i, u_i) <- (m_i, \ell_i, u_i) \bigoplus (\hat{m}_i, \hat{\ell}_i, \hat{u}_i)
 */
template <typename RowExpr, typename ConstRowExpr>
inline void combine_softmax_state(
    float& m_i,
    float& l_i,
    RowExpr&& u_i,
    float m_hat,
    float l_hat,
    const ConstRowExpr& u_hat
) {
    const float m_new = std::max(m_i, m_hat);

    const float alpha = (m_i == -std::numeric_limits<float>::infinity())
        ? 0.0f
        : std::exp(m_i - m_new);
    const float beta = (m_hat == -std::numeric_limits<float>::infinity())
        ? 0.0f
        : std::exp(m_hat - m_new);

    l_i = alpha * l_i + beta * l_hat;
    u_i.noalias() = alpha * u_i + beta * u_hat;
    m_i = m_new;
}

/**
 * Recursive cache-oblivious accumulator: DenseAccumulate(I, J, m, \ell, u)
 */
inline void dense_accumulate(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    IndexRange I,
    IndexRange J,
    Eigen::VectorXf& m,
    Eigen::VectorXf& l,
    AttentionMatrixf& u,
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

        // Compute \hat{m}_i and \hat{\ell}_i, transforming S_block in-place into exp(s_{ij} - \hat{m}_i)
        for (Eigen::Index r = 0; r < I.len; ++r) {
            auto s_row = S_block.row(r);
            const float m_hat_i = s_row.maxCoeff();
            m_hat(r) = m_hat_i;

            s_row = (s_row.array() - m_hat_i).exp();
            l_hat(r) = s_row.sum();
        }

        // Vectorized computation of \hat{u}_i <- \sum_{j \in J} exp(s_{ij} - \hat{m}_i) v_j
        // across all i \in I simultaneously via GEMM: \hat{U}_I = P_{I,J} * V_J
        auto u_hat_block = ws.u_hat.topRows(I.len);
        u_hat_block.noalias() = S_block * V_J;

        // Apply associative reduction: (m_i, \ell_i, u_i) <- (m_i, \ell_i, u_i) \bigoplus (\hat{m}_i, \hat{\ell}_i, \hat{u}_i)
        for (Eigen::Index r = 0; r < I.len; ++r) {
            const Eigen::Index i = I.start + r;
            combine_softmax_state(
                m(i),
                l(i),
                u.row(i),
                m_hat(r),
                l_hat(r),
                u_hat_block.row(r)
            );
        }
        return;
    }

    // Recursive Step: Split I = I_1 \cup I_2, J = J_1 \cup J_2 into consecutive halves
    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    const IndexRange I1{I.start, i_mid};
    const IndexRange I2{I.start + i_mid, I.len - i_mid};

    const IndexRange J1{J.start, j_mid};
    const IndexRange J2{J.start + j_mid, J.len - j_mid};

    // Quadrant 1: (I_1, J_1)
    dense_accumulate(Q, K, V, I1, J1, m, l, u, b, scale, ws);
    // Quadrant 2: (I_1, J_2)
    dense_accumulate(Q, K, V, I1, J2, m, l, u, b, scale, ws);
    // Quadrant 3: (I_2, J_1)
    dense_accumulate(Q, K, V, I2, J1, m, l, u, b, scale, ws);
    // Quadrant 4: (I_2, J_2)
    dense_accumulate(Q, K, V, I2, J2, m, l, u, b, scale, ws);
}

} // namespace detail

/**
 * Cache-Oblivious Recursive Attention: DenseAttention(Q, K, V)
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

    // 1. Initialize (m_i, \ell_i, u_i) <- (-\infty, 0, 0) for i = 0, ..., N - 1
    Eigen::VectorXf m = Eigen::VectorXf::Constant(N, -std::numeric_limits<float>::infinity());
    Eigen::VectorXf l = Eigen::VectorXf::Zero(N);
    AttentionMatrixf u = AttentionMatrixf::Zero(N, 64);

    // Preallocate leaf workspace once (sized to base case threshold b)
    detail::LeafWorkspace ws(b);

    // 2. Recursive divide-and-conquer accumulation over [0, N) x [0, N)
    detail::dense_accumulate(
        Q, K, V,
        IndexRange{0, N},
        IndexRange{0, N},
        m, l, u,
        b, scale, ws
    );

    // 3. Final row-wise normalization: O_{i,:} <- u_i / \ell_i
    AttentionMatrixf O(N, 64);
    for (Eigen::Index i = 0; i < N; ++i) {
        if (l(i) > 0.0f) {
            O.row(i).noalias() = u.row(i) / l(i);
        } else {
            O.row(i).setZero();
        }
    }

    return DenseAttentionResult{std::move(O), std::move(m), std::move(l)};
}
