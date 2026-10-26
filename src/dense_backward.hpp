#pragma once

#include "attention_utilities.hpp"
#include "dense_accumulate.hpp" // IndexRange
#include <cilk/cilk.h>
#include <cmath>
#include <limits>
#include <utility>

/**
 * Gradient buffer \Gamma = (\Delta Q, \Delta K, \Delta V) over a query interval I and key
 * interval J. Rows are addressed by global index; each buffer records its own origins, so
 * private children and the shared root can be merged with the original indices.
 */
struct GradientSummary {
    Eigen::Index q_offset = 0;
    Eigen::Index k_offset = 0;
    AttentionMatrixf dQ; // |I| x d
    AttentionMatrixf dK; // |J| x d
    AttentionMatrixf dV; // |J| x d

    GradientSummary() = default;

    /// AllocateGradients(I, J, d): uninitialized arrays with |I|, |J|, |J| rows
    GradientSummary(IndexRange I, IndexRange J)
        : q_offset(I.start),
          k_offset(J.start),
          dQ(I.len, 64),
          dK(J.len, 64),
          dV(J.len, 64) {}

    [[nodiscard]] auto dQ_rows(IndexRange I) { return dQ.middleRows(I.start - q_offset, I.len); }
    [[nodiscard]] auto dK_rows(IndexRange J) { return dK.middleRows(J.start - k_offset, J.len); }
    [[nodiscard]] auto dV_rows(IndexRange J) { return dV.middleRows(J.start - k_offset, J.len); }
};

/// Which gradients a traversal writes: all three, \Delta Q only, or (\Delta K, \Delta V) only
enum class GradientMode { All, Q, KV };

/**
 * Read-only inputs and saved statistics shared by every recursive call.
 * L is the forward row-wise log-sum-exp, so e^{s - m_i} / \ell_i = e^{s - L_i}.
 */
struct BackwardInputs {
    const AttentionMatrixf& Q;
    const AttentionMatrixf& K;
    const AttentionMatrixf& V;
    const AttentionMatrixf& dO;
    const Eigen::VectorXf& L;
    const Eigen::VectorXf& D; // D_i = dO_{i,:} O_{i,:}^T
    float scale;              // 1 / sqrt(d)
};

namespace detail {

using RowMajorScratch = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * Preallocated b x b scratch for leaf evaluations (one per serial traversal).
 */
struct BackwardLeafWorkspace {
    RowMajorScratch S;  // Scores, then probabilities P in place
    RowMajorScratch dP; // dO V^T, then score gradients dS in place

    explicit BackwardLeafWorkspace(Eigen::Index b) : S(b, b), dP(b, b) {}
};

/**
 * Blocked leaf: BackwardPair(i, j, \Gamma) for all (i, j) \in I x J at once, as dense products
 * (the "fixed leaf microkernel" form of Algorithm 5):
 *   P = exp(Q_I K_J^T / sqrt(d) - L_I),  dS = P \odot (dO_I V_J^T - D_I)
 *   \Delta V_J += P^T dO_I,  \Delta Q_I += dS K_J / sqrt(d),  \Delta K_J += dS^T Q_I / sqrt(d)
 * mode restricts the writes to the gradients this traversal owns.
 */
inline void backward_leaf(
    const BackwardInputs& in,
    IndexRange I,
    IndexRange J,
    GradientSummary& gamma,
    GradientMode mode,
    BackwardLeafWorkspace& ws
) {
    auto Q_I = in.Q.middleRows(I.start, I.len);
    auto dO_I = in.dO.middleRows(I.start, I.len);
    auto K_J = in.K.middleRows(J.start, J.len);
    auto V_J = in.V.middleRows(J.start, J.len);

    auto P = ws.S.topLeftCorner(I.len, J.len);
    auto dS = ws.dP.topLeftCorner(I.len, J.len);

    // a = e^{s - m_i} / \ell_i = e^{s - L_i}
    P.noalias() = (Q_I * K_J.transpose()) * in.scale;
    for (Eigen::Index r = 0; r < I.len; ++r) {
        const float lse = in.L(I.start + r);
        if (std::isfinite(lse)) {
            P.row(r) = (P.row(r).array() - lse).exp();
        } else {
            P.row(r).setZero();
        }
    }

    if (mode != GradientMode::Q) {
        gamma.dV_rows(J).noalias() += P.transpose() * dO_I;
    }

    // h = a (dO_i v_j^T - D_i)
    dS.noalias() = dO_I * V_J.transpose();
    for (Eigen::Index r = 0; r < I.len; ++r) {
        dS.row(r).array() = P.row(r).array() * (dS.row(r).array() - in.D(I.start + r));
    }

    if (mode != GradientMode::KV) {
        gamma.dQ_rows(I).noalias() += (dS * K_J) * in.scale;
    }
    if (mode != GradientMode::Q) {
        gamma.dK_rows(J).noalias() += (dS.transpose() * Q_I) * in.scale;
    }
}

/**
 * Serial cache-oblivious gradient accumulation: BackwardAccumulate(I, J, \Gamma) (Algorithm 6).
 * Adds the (I, J) contribution to the gradients selected by mode; \Gamma must be initialized.
 */
inline void backward_accumulate(
    const BackwardInputs& in,
    IndexRange I,
    IndexRange J,
    GradientSummary& gamma,
    GradientMode mode,
    Eigen::Index b,
    BackwardLeafWorkspace& ws
) {
    if (I.len <= 0 || J.len <= 0) {
        return;
    }

    if (I.len <= b && J.len <= b) {
        backward_leaf(in, I, J, gamma, mode, ws);
        return;
    }

    // Split I = I_1 \cup I_2, J = J_1 \cup J_2 into consecutive halves; visit in lexicographic order
    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    const IndexRange I1{I.start, i_mid};
    const IndexRange I2{I.start + i_mid, I.len - i_mid};
    const IndexRange J1{J.start, j_mid};
    const IndexRange J2{J.start + j_mid, J.len - j_mid};

    backward_accumulate(in, I1, J1, gamma, mode, b, ws);
    backward_accumulate(in, I1, J2, gamma, mode, b, ws);
    backward_accumulate(in, I2, J1, gamma, mode, b, ws);
    backward_accumulate(in, I2, J2, gamma, mode, b, ws);
}

/**
 * parallel for (i, k) \in I x [d]: \Gamma.\Delta Q_{i,k} <- 0
 * parallel for (j, k) \in J x [d]: (\Gamma.\Delta K_{j,k}, \Gamma.\Delta V_{j,k}) <- (0, 0)
 *
 * noinline: cilk_for loops inlined into the spawning recursions below can crash OpenCilk 3.0's
 * jump-threading pass with AVX-512 targets (see parallel_forward_merge).
 */
__attribute__((noinline)) inline void reset_gradients(GradientSummary& gamma, IndexRange I, IndexRange J) {
    cilk_for (Eigen::Index r = 0; r < I.len; ++r) {
        gamma.dQ.row(I.start - gamma.q_offset + r).setZero();
    }
    cilk_for (Eigen::Index c = 0; c < J.len; ++c) {
        gamma.dK.row(J.start - gamma.k_offset + c).setZero();
        gamma.dV.row(J.start - gamma.k_offset + c).setZero();
    }
}

/**
 * ParallelGradientMerge(I, J, \Gamma, \Gamma_{11}, \Gamma_{12}, \Gamma_{21}, \Gamma_{22}) (Algorithm 9):
 * query gradients add across key halves, key/value gradients add across query halves.
 * noinline for the same compiler workaround as reset_gradients.
 */
__attribute__((noinline)) inline void parallel_gradient_merge(
    IndexRange I,
    IndexRange J,
    GradientSummary& gamma,
    const GradientSummary& gamma_11,
    const GradientSummary& gamma_12,
    const GradientSummary& gamma_21,
    const GradientSummary& gamma_22
) {
    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    // parallel for a \in {1, 2}, i \in I_a: \Delta Q_i <- \Gamma_{a1}.\Delta Q_i + \Gamma_{a2}.\Delta Q_i
    cilk_for (Eigen::Index r = 0; r < I.len; ++r) {
        const Eigen::Index i = I.start + r;
        const GradientSummary& a1 = (r < i_mid) ? gamma_11 : gamma_21;
        const GradientSummary& a2 = (r < i_mid) ? gamma_12 : gamma_22;
        gamma.dQ.row(i - gamma.q_offset) = a1.dQ.row(i - a1.q_offset) + a2.dQ.row(i - a2.q_offset);
    }

    // parallel for b' \in {1, 2}, j \in J_{b'}: \Delta K_j, \Delta V_j <- \Gamma_{1b'} + \Gamma_{2b'}
    cilk_for (Eigen::Index c = 0; c < J.len; ++c) {
        const Eigen::Index j = J.start + c;
        const GradientSummary& b1 = (c < j_mid) ? gamma_11 : gamma_12;
        const GradientSummary& b2 = (c < j_mid) ? gamma_21 : gamma_22;
        gamma.dK.row(j - gamma.k_offset) = b1.dK.row(j - b1.k_offset) + b2.dK.row(j - b2.k_offset);
        gamma.dV.row(j - gamma.k_offset) = b1.dV.row(j - b1.k_offset) + b2.dV.row(j - b2.k_offset);
    }
}

/// parallel for i \in [N]: D_i <- dO_{i,:} O_{i,:}^T
inline Eigen::VectorXf output_gradient_dots(const AttentionMatrixf& dO, const AttentionMatrixf& O) {
    Eigen::VectorXf D(O.rows());
    cilk_for (Eigen::Index i = 0; i < O.rows(); ++i) {
        D(i) = dO.row(i).dot(O.row(i));
    }
    return D;
}

inline AttentionGradients take_gradients(GradientSummary& gamma) {
    return AttentionGradients{std::move(gamma.dQ), std::move(gamma.dK), std::move(gamma.dV)};
}

} // namespace detail
