#pragma once

// Mixed sparse backward with query and key-value ownership (Algorithms 28, 29, 32-34): gradients
// under a structural mask (quadtree) with exactly zero query rows, jointly zero key/value rows and
// zero upstream-gradient rows. Two traversals: mode Q writes only dQ, mode KV only dK and dV.
//
// Every closed form uses a_i = e^{-m_i} / l_i = e^{-L_i}, the probability of a zero score under the
// saved forward normalizers. As in the sparse forward, leaves are blocked (-inf masking; per pair
// when at most 1/8 of a leaf is allowed) and a grain cutoff runs small owned intervals serially, so
// dense, fully active inputs follow the ParallelOwnedBackward schedule.

#include "dense_backward.hpp"
#include "mask_quadtree.hpp"
#include "row_activity.hpp"
#include "sparse_forward.hpp" // SparseShortcuts, split_quadrants, is_leaf
#include <cilk/cilk.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace detail {

struct SparseBackwardContext {
    BackwardInputs in; // Q, K, V, dO, L, D, scale
    const MaskQuadtree& mask;
    const RowActivity& q_active;
    const RowActivity& kv_active;
    const RowActivity& do_active;
    Eigen::Index b;
};

/// a_i = e^{-L_i}: the probability of a zero score in row i (0 for an empty row)
inline float zero_score_probability(const SparseBackwardContext& ctx, Eigen::Index i) {
    const float lse = ctx.in.L(i);
    return std::isfinite(lse) ? std::exp(-lse) : 0.0f;
}

/// Calls f(i, j) for every allowed pair of a mask block, depth first (StreamAllowed)
template <typename F>
inline void for_each_allowed(const SparseBackwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J, F&& f) {
    if (h == kMaskEmpty || I.len <= 0 || J.len <= 0) {
        return;
    }
    if (h == kMaskFull) {
        for (Eigen::Index i = I.start; i < I.start + I.len; ++i) {
            for (Eigen::Index j = J.start; j < J.start + J.len; ++j) {
                f(i, j);
            }
        }
        return;
    }
    if (is_leaf(I, J, ctx.b)) {
        const uint64_t* bits = ctx.mask.leaf_bits(h);
        for (Eigen::Index r = 0; r < I.len; ++r) {
            for (Eigen::Index c = 0; c < J.len; ++c) {
                if (MaskQuadtree::bit(bits, r * J.len + c)) {
                    f(I.start + r, J.start + c);
                }
            }
        }
        return;
    }
    IndexRange qI[4], qJ[4];
    split_quadrants(I, J, qI, qJ);
    for (int q = 0; q < 4; ++q) {
        for_each_allowed(ctx, ctx.mask.child(h, q), qI[q], qJ[q], f);
    }
}

/// One allowed pair's contribution to the gradients this traversal owns (OwnedBackward({i}, {j}))
inline void backward_pair(const SparseBackwardContext& ctx, Eigen::Index i, Eigen::Index j, GradientMode mode,
                          GradientSummary& gamma) {
    const BackwardInputs& in = ctx.in;
    const float lse = in.L(i);
    if (!std::isfinite(lse)) {
        return;
    }
    const float a = std::exp(in.Q.row(i).dot(in.K.row(j)) * in.scale - lse);
    const float h = a * (in.dO.row(i).dot(in.V.row(j)) - in.D(i));
    if (mode == GradientMode::Q) {
        gamma.dQ.row(i - gamma.q_offset).noalias() += (h * in.scale) * in.K.row(j);
    } else {
        gamma.dV.row(j - gamma.k_offset).noalias() += a * in.dO.row(i);
        gamma.dK.row(j - gamma.k_offset).noalias() += (h * in.scale) * in.Q.row(i);
    }
}

/// A b x b leaf: the blocked kernel of BackwardAccumulate with disallowed pairs masked, or per pair
inline void masked_backward_leaf(const SparseBackwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J,
                                 GradientMode mode, GradientSummary& gamma, BackwardLeafWorkspace& ws) {
    const int64_t allowed = ctx.mask.count(h, I, J);
    const int64_t area = static_cast<int64_t>(I.len) * J.len;
    const uint64_t* bits = (h >= 0) ? ctx.mask.leaf_bits(h) : nullptr;

    if (bits != nullptr && allowed * 8 <= area) {
        for (Eigen::Index r = 0; r < I.len; ++r) {
            const Eigen::Index i = I.start + r;
            if (!ctx.do_active.active(i)) {
                continue;
            }
            for (Eigen::Index c = 0; c < J.len; ++c) {
                if (MaskQuadtree::bit(bits, r * J.len + c)) {
                    backward_pair(ctx, i, J.start + c, mode, gamma);
                }
            }
        }
        return;
    }

    if (bits == nullptr) {
        backward_leaf(ctx.in, I, J, gamma, mode, ws); // Full leaf: the dense kernel
        return;
    }

    const BackwardInputs& in = ctx.in;
    auto Q_I = in.Q.middleRows(I.start, I.len);
    auto dO_I = in.dO.middleRows(I.start, I.len);
    auto K_J = in.K.middleRows(J.start, J.len);
    auto V_J = in.V.middleRows(J.start, J.len);
    auto P = ws.S.topLeftCorner(I.len, J.len);
    auto dS = ws.dP.topLeftCorner(I.len, J.len);

    // P = exp(S - L) on allowed pairs, 0 elsewhere
    P.noalias() = (Q_I * K_J.transpose()) * in.scale;
    for (Eigen::Index r = 0; r < I.len; ++r) {
        const float lse = in.L(I.start + r);
        if (!std::isfinite(lse)) {
            P.row(r).setZero();
            continue;
        }
        P.row(r) = (P.row(r).array() - lse).exp();
        for (Eigen::Index c = 0; c < J.len; ++c) {
            if (!MaskQuadtree::bit(bits, r * J.len + c)) {
                P(r, c) = 0.0f;
            }
        }
    }
    if (mode != GradientMode::Q) {
        gamma.dV_rows(J).noalias() += P.transpose() * dO_I;
    }
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

/// Equation (145): the zero-query moment aggregate is cheaper than pair work
inline bool zero_q_moment_pays(IndexRange I, IndexRange J) {
    constexpr Eigen::Index d = 64;
    return (I.len + J.len) * d * d < I.len * J.len * d;
}

/**
 * Serial mixed backward over I x J restricted to the gradients `mode` owns: Algorithm 28's case
 * analysis with Algorithm 34's owned terminal contributions. Prunes mask-empty and zero-dO blocks.
 */
inline void mixed_backward_accumulate(const SparseBackwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J,
                                      GradientMode mode, GradientSummary& gamma, BackwardLeafWorkspace& ws) {
    const int64_t c = ctx.mask.count(h, I, J);
    if (c == 0 || I.len <= 0 || J.len <= 0 || ctx.do_active.count(I) == 0) {
        return;
    }
    const BackwardInputs& in = ctx.in;
    const int64_t area = static_cast<int64_t>(I.len) * J.len;
    const Eigen::Index c_q = ctx.q_active.count(I);
    const Eigen::Index c_kv = ctx.kv_active.count(J);

    if (c == area) {
        if (c_kv == 0) {
            // Equation (142): g_K = -(1/sqrt d) sum_i a_i D_i q_i, g_V = sum_i a_i dO_i; no dQ
            if (mode == GradientMode::KV) {
                Eigen::Matrix<float, 1, 64> g_K = Eigen::Matrix<float, 1, 64>::Zero();
                Eigen::Matrix<float, 1, 64> g_V = Eigen::Matrix<float, 1, 64>::Zero();
                for (Eigen::Index i = I.start; i < I.start + I.len; ++i) {
                    const float a = zero_score_probability(ctx, i);
                    g_K.noalias() -= (a * in.D(i) * in.scale) * in.Q.row(i);
                    g_V.noalias() += a * in.dO.row(i);
                }
                gamma.dK_rows(J).rowwise() += g_K;
                gamma.dV_rows(J).rowwise() += g_V;
            }
            return;
        }
        if (c_q == 0 && zero_q_moment_pays(I, J)) {
            if (mode == GradientMode::Q) {
                // Equation (144): dQ_i += (a_i / sqrt d)(dO_i C_J - D_i k_sum), C_J = sum_j v_j^T k_j
                const Eigen::Matrix<float, 64, 64> C_J =
                    in.V.middleRows(J.start, J.len).transpose() * in.K.middleRows(J.start, J.len);
                const Eigen::Matrix<float, 1, 64> k_sum = in.K.middleRows(J.start, J.len).colwise().sum();
                for (Eigen::Index i = I.start; i < I.start + I.len; ++i) {
                    const float a = zero_score_probability(ctx, i) * in.scale;
                    gamma.dQ.row(i - gamma.q_offset).noalias() += a * (in.dO.row(i) * C_J - in.D(i) * k_sum);
                }
            } else {
                // dV_j += sum_i a_i dO_i for every j; no dK (q_i = 0)
                Eigen::Matrix<float, 1, 64> g = Eigen::Matrix<float, 1, 64>::Zero();
                for (Eigen::Index i = I.start; i < I.start + I.len; ++i) {
                    g.noalias() += zero_score_probability(ctx, i) * in.dO.row(i);
                }
                gamma.dV_rows(J).rowwise() += g;
            }
            return;
        }
        if (c_q == I.len && c_kv == J.len && ctx.do_active.count(I) == I.len) {
            backward_accumulate(in, I, J, gamma, mode, ctx.b, ws);
            return;
        }
    } else if (c_kv == 0 || c_q == 0) {
        // MaskedZeroBackward (Algorithm 29) restricted to the owned gradients
        if (c_kv == 0 && mode == GradientMode::Q) {
            return; // Zero keys contribute no query gradient
        }
        for_each_allowed(ctx, h, I, J, [&](Eigen::Index i, Eigen::Index j) {
            if (!ctx.do_active.active(i)) {
                return;
            }
            const float a = zero_score_probability(ctx, i);
            if (c_kv == 0) {
                gamma.dV.row(j - gamma.k_offset).noalias() += a * in.dO.row(i);
                gamma.dK.row(j - gamma.k_offset).noalias() -= (a * in.D(i) * in.scale) * in.Q.row(i);
            } else if (mode == GradientMode::Q) {
                const float hv = a * (in.dO.row(i).dot(in.V.row(j)) - in.D(i));
                gamma.dQ.row(i - gamma.q_offset).noalias() += (hv * in.scale) * in.K.row(j);
            } else {
                gamma.dV.row(j - gamma.k_offset).noalias() += a * in.dO.row(i);
            }
        });
        return;
    }

    if (is_leaf(I, J, ctx.b)) {
        masked_backward_leaf(ctx, h, I, J, mode, gamma, ws);
        return;
    }

    IndexRange qI[4], qJ[4];
    split_quadrants(I, J, qI, qJ);
    for (int q = 0; q < 4; ++q) {
        mixed_backward_accumulate(ctx, ctx.mask.child(h, q), qI[q], qJ[q], mode, gamma, ws);
    }
}

/// Terminal predicate B0(X): a full aggregate, a partial-mask zero-row kernel, or a fixed leaf
inline bool backward_terminal(const SparseBackwardContext& ctx, IndexRange I, IndexRange J, int64_t c) {
    const int64_t area = static_cast<int64_t>(I.len) * J.len;
    const Eigen::Index c_q = ctx.q_active.count(I);
    const Eigen::Index c_kv = ctx.kv_active.count(J);
    if (c == area && (c_kv == 0 || (c_q == 0 && zero_q_moment_pays(I, J)))) {
        return true;
    }
    if (c > 0 && c < area && (c_kv == 0 || c_q == 0)) {
        return true;
    }
    return is_leaf(I, J, ctx.b);
}

/**
 * OwnedMixedBackward(I, J, mode, Gamma) (Algorithm 33). Mode Q runs the two query halves in
 * parallel against each key half in turn; mode KV runs the two key halves in parallel against each
 * query half in turn, so concurrent tasks write disjoint rows of the shared Gamma.
 */
inline void owned_mixed_backward(const SparseBackwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J,
                                 GradientMode mode, GradientSummary& gamma, Eigen::Index grain,
                                 BackwardLeafWorkspaces& workspaces) {
    const int64_t c = ctx.mask.count(h, I, J);
    if (c == 0 || I.len <= 0 || J.len <= 0 || ctx.do_active.count(I) == 0) {
        return;
    }
    const Eigen::Index owned_len = (mode == GradientMode::Q) ? I.len : J.len;
    if (owned_len <= grain || backward_terminal(ctx, I, J, c)) {
        BackwardLeafWorkspace& ws = workspaces; // This worker's view; no spawn or sync follows
        ws.reserve(ctx.b);
        mixed_backward_accumulate(ctx, h, I, J, mode, gamma, ws);
        return;
    }

    IndexRange qI[4], qJ[4];
    split_quadrants(I, J, qI, qJ);
    if (mode == GradientMode::Q) {
        cilk_spawn owned_mixed_backward(ctx, ctx.mask.child(h, 0), qI[0], qJ[0], mode, gamma, grain, workspaces);
                   owned_mixed_backward(ctx, ctx.mask.child(h, 2), qI[2], qJ[2], mode, gamma, grain, workspaces);
        cilk_sync;
        cilk_spawn owned_mixed_backward(ctx, ctx.mask.child(h, 1), qI[1], qJ[1], mode, gamma, grain, workspaces);
                   owned_mixed_backward(ctx, ctx.mask.child(h, 3), qI[3], qJ[3], mode, gamma, grain, workspaces);
        cilk_sync;
    } else {
        cilk_spawn owned_mixed_backward(ctx, ctx.mask.child(h, 0), qI[0], qJ[0], mode, gamma, grain, workspaces);
                   owned_mixed_backward(ctx, ctx.mask.child(h, 1), qI[1], qJ[1], mode, gamma, grain, workspaces);
        cilk_sync;
        cilk_spawn owned_mixed_backward(ctx, ctx.mask.child(h, 2), qI[2], qJ[2], mode, gamma, grain, workspaces);
                   owned_mixed_backward(ctx, ctx.mask.child(h, 3), qI[3], qJ[3], mode, gamma, grain, workspaces);
        cilk_sync;
    }
}

} // namespace detail

/**
 * OwnedMixedBackwardAttention (Algorithm 32): activity discovery and prefix counts for Q, K/V and
 * dO, D, zeroed gradients, then the Q-mode and KV-mode traversals. O and L must come from the
 * masked forward with the same mask and inputs.
 */
inline AttentionGradients owned_mixed_backward_attention(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    const AttentionMatrixf& O,
    const AttentionMatrixf& dO,
    const Eigen::VectorXf& L,
    const MaskQuadtree& mask,
    SparseShortcuts shortcuts,
    Eigen::Index b = 64,
    Eigen::Index grain = 256
) {
    const Eigen::Index N = Q.rows();
    if (mask.n != N || mask.b != b) {
        throw std::runtime_error("Mask quadtree was built for n = " + std::to_string(mask.n) + ", b = " +
                                 std::to_string(mask.b) + " but the solve uses n = " + std::to_string(N) +
                                 ", b = " + std::to_string(b));
    }
    const RowActivity q_active = shortcuts.zero_q ? RowActivity::from_rows({&Q}) : RowActivity::all_active(N);
    const RowActivity kv_active = shortcuts.zero_kv ? RowActivity::from_rows({&K, &V}) : RowActivity::all_active(N);
    const RowActivity do_active = shortcuts.zero_do ? RowActivity::from_rows({&dO}) : RowActivity::all_active(N);
    const Eigen::VectorXf D = detail::output_gradient_dots(dO, O);
    const detail::SparseBackwardContext ctx{
        BackwardInputs{Q, K, V, dO, L, D, 1.0f / std::sqrt(static_cast<float>(Q.cols()))},
        mask, q_active, kv_active, do_active, b};
    const IndexRange full_range{0, N};

    GradientSummary gamma(full_range, full_range);
    detail::reset_gradients(gamma, full_range, full_range);
    detail::BackwardLeafWorkspaces workspaces;
    detail::owned_mixed_backward(ctx, mask.root, full_range, full_range, GradientMode::Q, gamma, grain, workspaces);
    detail::owned_mixed_backward(ctx, mask.root, full_range, full_range, GradientMode::KV, gamma, grain, workspaces);
    return detail::take_gradients(gamma);
}
