#pragma once

// Query-owned parallel mixed forward (Algorithms 15-17 and 19): attention under a structural mask
// supplied as a quadtree, with exactly zero query rows and jointly zero key/value rows.
//
// Departures from the pseudocode, both chosen so that a fully dense, fully active input runs the
// same schedule and kernels as the dense QueryParallel solver:
//   * Leaves are blocked: a b x b leaf computes Q_I K_J^T as one product and sets disallowed
//     scores to -inf, instead of streaming allowed pairs one at a time. Leaves with at most 1/8 of
//     their pairs allowed use per-pair evaluation, where the blocked product would waste work.
//   * A grain cutoff: once |I| <= grain the owned recursion runs the serial traversal
//     (MixedForwardAccumulate) on the block, as QueryParallel does, instead of spawning to leaves.

#include "dense_accumulate.hpp"
#include "mask_quadtree.hpp"
#include "row_activity.hpp"
#include <cilk/cilk.h>
#include <cilk/holder.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

/// Which numerical shortcuts are enabled (Algorithm 17's form); disabled ones treat rows as active
struct SparseShortcuts {
    bool zero_q = true;
    bool zero_kv = true;
    bool zero_do = true; // Backward only
};

struct SparseForwardResult {
    AttentionMatrixf O;
    Eigen::VectorXf L; // Row-wise log-sum-exp m_i + log(l_i), for the backward pass
};

namespace detail {

struct SparseForwardContext {
    const AttentionMatrixf& Q;
    const AttentionMatrixf& K;
    const AttentionMatrixf& V;
    const MaskQuadtree& mask;
    const RowActivity& q_active;
    const RowActivity& kv_active;
    Eigen::Index b;
    float scale;
};

/// Per-worker scratch: the dense leaf workspace plus a column list for per-pair leaves
struct SparseForwardWorkspace {
    LeafWorkspace leaf;
    std::vector<Eigen::Index> cols;

    SparseForwardWorkspace() = default;
    void reserve(Eigen::Index b) {
        leaf.reserve(b);
        if (static_cast<Eigen::Index>(cols.size()) < b) {
            cols.resize(b);
        }
    }
};

using SparseForwardWorkspaces = cilk::holder<SparseForwardWorkspace>;

inline void split_quadrants(IndexRange I, IndexRange J, IndexRange (&qI)[4], IndexRange (&qJ)[4]) {
    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;
    const IndexRange I1{I.start, i_mid}, I2{I.start + i_mid, I.len - i_mid};
    const IndexRange J1{J.start, j_mid}, J2{J.start + j_mid, J.len - j_mid};
    qI[0] = I1; qJ[0] = J1;
    qI[1] = I1; qJ[1] = J2;
    qI[2] = I2; qJ[2] = J1;
    qI[3] = I2; qJ[3] = J2;
}

inline bool is_leaf(IndexRange I, IndexRange J, Eigen::Index b) { return I.len <= b && J.len <= b; }

/// Sigma_i <- Sigma_i (+) (0, mass, u) for a contribution whose scores are all zero
template <typename Row>
inline void combine_zero_score(AttentionSummary& sigma, Eigen::Index i, float mass, const Row& u) {
    combine_softmax_state(sigma.m_at(i), sigma.l_at(i), sigma.U_row(i), sigma.m_at(i), sigma.l_at(i),
                          sigma.U_row(i), 0.0f, mass, u);
}

/**
 * StreamAllowed aggregated per row over a mask block: t[r] += allowed keys of row I.start + r,
 * and, when g is given, g.row(r) += the sum of their value rows. Full descendants add |J| and a
 * value-column sum; partial leaves read their bitmap.
 */
inline void accumulate_allowed_rows(const SparseForwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J,
                                    Eigen::Index row0, std::vector<float>& t, AttentionMatrixf* g) {
    if (h == kMaskEmpty || I.len <= 0 || J.len <= 0) {
        return;
    }
    if (h == kMaskFull) {
        Eigen::Matrix<float, 1, 64> v_sum;
        if (g != nullptr) {
            v_sum = ctx.V.middleRows(J.start, J.len).colwise().sum();
        }
        for (Eigen::Index r = 0; r < I.len; ++r) {
            t[I.start - row0 + r] += static_cast<float>(J.len);
            if (g != nullptr) {
                g->row(I.start - row0 + r) += v_sum;
            }
        }
        return;
    }
    if (is_leaf(I, J, ctx.b)) {
        const uint64_t* bits = ctx.mask.leaf_bits(h);
        for (Eigen::Index r = 0; r < I.len; ++r) {
            for (Eigen::Index c = 0; c < J.len; ++c) {
                if (MaskQuadtree::bit(bits, r * J.len + c)) {
                    t[I.start - row0 + r] += 1.0f;
                    if (g != nullptr) {
                        g->row(I.start - row0 + r) += ctx.V.row(J.start + c);
                    }
                }
            }
        }
        return;
    }
    IndexRange qI[4], qJ[4];
    split_quadrants(I, J, qI, qJ);
    for (int q = 0; q < 4; ++q) {
        accumulate_allowed_rows(ctx, ctx.mask.child(h, q), qI[q], qJ[q], row0, t, g);
    }
}

/// A b x b leaf over its allowed pairs: blocked with -inf masking, or per pair when very sparse
inline void masked_forward_leaf(const SparseForwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J,
                                AttentionSummary& sigma, SparseForwardWorkspace& ws) {
    const int64_t allowed = ctx.mask.count(h, I, J);
    const int64_t area = static_cast<int64_t>(I.len) * J.len;
    const uint64_t* bits = (h >= 0) ? ctx.mask.leaf_bits(h) : nullptr;
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();

    if (bits != nullptr && allowed * 8 <= area) {
        // Per-pair evaluation: only allowed (i, j) scores are formed
        Eigen::Matrix<float, 1, 64> u;
        for (Eigen::Index r = 0; r < I.len; ++r) {
            Eigen::Index n_cols = 0;
            for (Eigen::Index c = 0; c < J.len; ++c) {
                if (MaskQuadtree::bit(bits, r * J.len + c)) {
                    ws.cols[n_cols++] = J.start + c;
                }
            }
            if (n_cols == 0) {
                continue;
            }
            const Eigen::Index i = I.start + r;
            float* s = ws.leaf.S.row(0).data(); // Scratch for this row's scores
            float m = kNegInf;
            for (Eigen::Index k = 0; k < n_cols; ++k) {
                s[k] = ctx.Q.row(i).dot(ctx.K.row(ws.cols[k])) * ctx.scale;
                m = std::max(m, s[k]);
            }
            float l = 0.0f;
            u.setZero();
            for (Eigen::Index k = 0; k < n_cols; ++k) {
                const float e = std::exp(s[k] - m);
                l += e;
                u.noalias() += e * ctx.V.row(ws.cols[k]);
            }
            combine_softmax_state(sigma.m_at(i), sigma.l_at(i), sigma.U_row(i), sigma.m_at(i), sigma.l_at(i),
                                  sigma.U_row(i), m, l, u);
        }
        return;
    }

    // Blocked evaluation, as in DenseAccumulate, with disallowed scores masked to -inf
    auto S = ws.leaf.S.topLeftCorner(I.len, J.len);
    S.noalias() = (ctx.Q.middleRows(I.start, I.len) * ctx.K.middleRows(J.start, J.len).transpose()) * ctx.scale;
    if (bits != nullptr) {
        for (Eigen::Index r = 0; r < I.len; ++r) {
            for (Eigen::Index c = 0; c < J.len; ++c) {
                if (!MaskQuadtree::bit(bits, r * J.len + c)) {
                    S(r, c) = kNegInf;
                }
            }
        }
    }
    for (Eigen::Index r = 0; r < I.len; ++r) {
        auto s_row = S.row(r);
        const float m = s_row.maxCoeff();
        ws.leaf.m_hat(r) = m;
        if (m == kNegInf) { // Row has no allowed key in this leaf
            s_row.setZero();
            ws.leaf.l_hat(r) = 0.0f;
            continue;
        }
        s_row = (s_row.array() - m).exp();
        ws.leaf.l_hat(r) = s_row.sum();
    }
    auto u_hat = ws.leaf.u_hat.topRows(I.len);
    u_hat.noalias() = S * ctx.V.middleRows(J.start, J.len);
    for (Eigen::Index r = 0; r < I.len; ++r) {
        const Eigen::Index i = I.start + r;
        combine_softmax_state(sigma.m_at(i), sigma.l_at(i), sigma.U_row(i), sigma.m_at(i), sigma.l_at(i),
                              sigma.U_row(i), ws.leaf.m_hat(r), ws.leaf.l_hat(r), u_hat.row(r));
    }
}

/**
 * MixedForwardAccumulate(I, J, Sigma) (Algorithm 15, with the zero-row kernels of Algorithm 16).
 * Serial; h is the mask block for I x J.
 */
inline void mixed_forward_accumulate(const SparseForwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J,
                                     AttentionSummary& sigma, SparseForwardWorkspace& ws) {
    const int64_t c = ctx.mask.count(h, I, J);
    if (c == 0 || I.len <= 0 || J.len <= 0) {
        return;
    }
    const int64_t area = static_cast<int64_t>(I.len) * J.len;
    const Eigen::Index c_q = ctx.q_active.count(I);
    const Eigen::Index c_kv = ctx.kv_active.count(J);

    if (c == area) {
        if (c_kv == 0) {
            // Every score is 0 and every value is 0: denominator mass only
            const Eigen::Matrix<float, 1, 64> zero = Eigen::Matrix<float, 1, 64>::Zero();
            for (Eigen::Index r = 0; r < I.len; ++r) {
                combine_zero_score(sigma, I.start + r, static_cast<float>(J.len), zero);
            }
            return;
        }
        if (c_q == 0) {
            // Every score is 0: uniform over J, numerator g_J = sum of v_j
            const Eigen::Matrix<float, 1, 64> g = ctx.V.middleRows(J.start, J.len).colwise().sum();
            for (Eigen::Index r = 0; r < I.len; ++r) {
                combine_zero_score(sigma, I.start + r, static_cast<float>(J.len), g);
            }
            return;
        }
        if (c_q == I.len && c_kv == J.len) {
            dense_accumulate(ctx.Q, ctx.K, ctx.V, I, J, sigma, ctx.b, ctx.scale, ws.leaf);
            return;
        }
    } else {
        if (c_kv == 0 && I.len <= c) {
            // MaskedZeroKVForward: per-row allowed counts
            std::vector<float> t(I.len, 0.0f);
            accumulate_allowed_rows(ctx, h, I, J, I.start, t, nullptr);
            const Eigen::Matrix<float, 1, 64> zero = Eigen::Matrix<float, 1, 64>::Zero();
            for (Eigen::Index r = 0; r < I.len; ++r) {
                if (t[r] > 0.0f) {
                    combine_zero_score(sigma, I.start + r, t[r], zero);
                }
            }
            return;
        }
        if (c_q == 0) {
            // MaskedZeroQForward: per-row allowed counts and allowed value sums
            std::vector<float> t(I.len, 0.0f);
            AttentionMatrixf g = AttentionMatrixf::Zero(I.len, 64);
            accumulate_allowed_rows(ctx, h, I, J, I.start, t, &g);
            for (Eigen::Index r = 0; r < I.len; ++r) {
                if (t[r] > 0.0f) {
                    combine_zero_score(sigma, I.start + r, t[r], g.row(r));
                }
            }
            return;
        }
    }

    if (is_leaf(I, J, ctx.b)) {
        masked_forward_leaf(ctx, h, I, J, sigma, ws);
        return;
    }

    IndexRange qI[4], qJ[4];
    split_quadrants(I, J, qI, qJ);
    for (int q = 0; q < 4; ++q) {
        const MaskHandle child = ctx.mask.child(h, q);
        if (ctx.mask.count(child, qI[q], qJ[q]) > 0) {
            mixed_forward_accumulate(ctx, child, qI[q], qJ[q], sigma, ws);
        }
    }
}

/// Terminal predicate F(X): an aggregate, a partial-mask zero-row kernel, or a fixed leaf
inline bool forward_terminal(const SparseForwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J, int64_t c) {
    const int64_t area = static_cast<int64_t>(I.len) * J.len;
    const Eigen::Index c_q = ctx.q_active.count(I);
    const Eigen::Index c_kv = ctx.kv_active.count(J);
    if (c == area && (c_kv == 0 || c_q == 0)) {
        return true;
    }
    if (c > 0 && c < area && ((c_kv == 0 && I.len <= c) || c_q == 0)) {
        return true;
    }
    return is_leaf(I, J, ctx.b);
}

/**
 * OwnedMixedForward(I, J, Sigma) (Algorithm 19): two parallel waves over disjoint query halves,
 * pruning mask-empty blocks, stopping at F(X) or at the grain cutoff.
 */
inline void owned_mixed_forward(const SparseForwardContext& ctx, MaskHandle h, IndexRange I, IndexRange J,
                                AttentionSummary& sigma, Eigen::Index grain, SparseForwardWorkspaces& workspaces) {
    const int64_t c = ctx.mask.count(h, I, J);
    if (c == 0 || I.len <= 0 || J.len <= 0) {
        return;
    }
    if (I.len <= grain || forward_terminal(ctx, h, I, J, c)) {
        SparseForwardWorkspace& ws = workspaces; // This worker's view; no spawn or sync follows
        ws.reserve(ctx.b);
        mixed_forward_accumulate(ctx, h, I, J, sigma, ws);
        return;
    }

    IndexRange qI[4], qJ[4];
    split_quadrants(I, J, qI, qJ);
    // Wave 1: (I1, J1) and (I2, J1); wave 2: (I1, J2) and (I2, J2)
    cilk_spawn owned_mixed_forward(ctx, ctx.mask.child(h, 0), qI[0], qJ[0], sigma, grain, workspaces);
               owned_mixed_forward(ctx, ctx.mask.child(h, 2), qI[2], qJ[2], sigma, grain, workspaces);
    cilk_sync;
    cilk_spawn owned_mixed_forward(ctx, ctx.mask.child(h, 1), qI[1], qJ[1], sigma, grain, workspaces);
               owned_mixed_forward(ctx, ctx.mask.child(h, 3), qI[3], qJ[3], sigma, grain, workspaces);
    cilk_sync;
}

} // namespace detail

/**
 * Owned mixed sparse forward driver (Algorithm 17 with mode = owned): activity discovery and
 * prefix counts, Sigma initialization, OwnedMixedForward, normalization. Also returns the row
 * log-sum-exp L for the backward pass. Every mask row must be nonempty.
 *
 * @param b     Base case size; must equal the mask quadtree's leaf size
 * @param grain Query-interval size below which recursion continues serially
 */
inline SparseForwardResult owned_mixed_forward_attention(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
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
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));
    const IndexRange full_range{0, N};

    const RowActivity q_active = shortcuts.zero_q ? RowActivity::from_rows({&Q}) : RowActivity::all_active(N);
    const RowActivity kv_active = shortcuts.zero_kv ? RowActivity::from_rows({&K, &V}) : RowActivity::all_active(N);
    const detail::SparseForwardContext ctx{Q, K, V, mask, q_active, kv_active, b, scale};

    AttentionSummary sigma(full_range);
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        sigma.m(i) = -std::numeric_limits<float>::infinity();
        sigma.l(i) = 0.0f;
        sigma.U.row(i).setZero();
    }

    detail::SparseForwardWorkspaces workspaces;
    detail::owned_mixed_forward(ctx, mask.root, full_range, full_range, sigma, grain, workspaces);

    SparseForwardResult result{AttentionMatrixf(N, 64), Eigen::VectorXf(N)};
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        const float l_i = sigma.l(i);
        if (l_i > 0.0f) {
            result.O.row(i).noalias() = sigma.U.row(i) / l_i;
            result.L(i) = sigma.m(i) + std::log(l_i);
        } else {
            result.O.row(i).setZero();
            result.L(i) = -std::numeric_limits<float>::infinity();
        }
    }
    return result;
}
