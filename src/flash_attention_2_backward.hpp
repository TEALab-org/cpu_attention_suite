#pragma once

#include "attention_utilities.hpp"
#include <cilk/cilk.h>
#include <cilk/cilksan.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>

// ============================================================================
// Strictly Lock-Free FlashAttention-2 Backward (For Cilkscale / Cilksan)
// ============================================================================

/**
 * Lock-Free Series-Parallel DAG Variant of FlashAttention-2 Backward:
 *
 * Eliminates all mutexes/atomics by splitting the Q-accumulation and KV-accumulation
 * into two lock-free parallel loops. Ideal when running under OpenCilk's Cilksan
 * race detector or Cilkscale work-span profiler.
 *
 * L is the forward pass's row-wise log-sum-exp of the scaled scores (see flash_attention_2's
 * L_out), so P_ij = exp(s_ij - L_i) is recomputed without a second softmax reduction.
 */
inline AttentionGradients flash_attention_2_backward_lockfree_opencilk(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    const AttentionMatrixf& O,
    const AttentionMatrixf& dO,
    const Eigen::VectorXf& L,
    bool causal = false,
    Eigen::Index Br = 64,
    Eigen::Index Bc = 64
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));

    AttentionGradients grads{
        AttentionMatrixf(N, 64),
        AttentionMatrixf(N, 64),
        AttentionMatrixf(N, 64)
    };

    Eigen::VectorXf D(N);
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        D(i) = dO.row(i).dot(O.row(i));
    }

    const Eigen::Index num_q_blocks = (N + Br - 1) / Br;
    const Eigen::Index num_kv_blocks = (N + Bc - 1) / Bc;

    // Sweep 1 (Parallel over K/V blocks): Exclusively computes dK and dV
    cilk_for (Eigen::Index bj = 0; bj < num_kv_blocks; ++bj) {
        const Eigen::Index c_start = bj * Bc;
        const Eigen::Index bc = std::min(Bc, N - c_start);
        const Eigen::Index c_end = c_start + bc;

        auto K_block = K.middleRows(c_start, bc);
        auto V_block = V.middleRows(c_start, bc);

        AttentionMatrixf dK_acc = AttentionMatrixf::Zero(bc, 64);
        AttentionMatrixf dV_acc = AttentionMatrixf::Zero(bc, 64);

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S_block(Br, bc);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P_block(Br, bc);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dP_block(Br, bc);

        const Eigen::Index bi_start = causal ? (c_start / Br) : 0;
        for (Eigen::Index bi = bi_start; bi < num_q_blocks; ++bi) {
            const Eigen::Index r_start = bi * Br;
            const Eigen::Index br = std::min(Br, N - r_start);

            auto Q_block = Q.middleRows(r_start, br);
            auto dO_block = dO.middleRows(r_start, br);

            auto S_sub = S_block.topRows(br);
            auto P_sub = P_block.topRows(br);
            auto dP_sub = dP_block.topRows(br);

            S_sub.noalias() = (Q_block * K_block.transpose()) * scale;

            if (causal && c_end > r_start) {
                for (Eigen::Index r = 0; r < br; ++r) {
                    const Eigen::Index gr = r_start + r;
                    for (Eigen::Index c = 0; c < bc; ++c) {
                        if (c_start + c > gr) {
                            S_sub(r, c) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }

            for (Eigen::Index r = 0; r < br; ++r) {
                const float lse = L(r_start + r);
                if (std::isfinite(lse)) {
                    P_sub.row(r) = (S_sub.row(r).array() - lse).exp();
                } else {
                    P_sub.row(r).setZero();
                }
            }

            dV_acc.noalias() += P_sub.transpose() * dO_block;
            dP_sub.noalias() = dO_block * V_block.transpose();

            for (Eigen::Index r = 0; r < br; ++r) {
                S_sub.row(r).array() = P_sub.row(r).array() * (dP_sub.row(r).array() - D(r_start + r));
            }

            dK_acc.noalias() += (S_sub.transpose() * Q_block) * scale;
        }

        grads.dK.middleRows(c_start, bc).noalias() = dK_acc;
        grads.dV.middleRows(c_start, bc).noalias() = dV_acc;
    }

    // Sweep 2 (Parallel over Q blocks): Exclusively computes dQ with deterministic reduction order
    cilk_for (Eigen::Index bi = 0; bi < num_q_blocks; ++bi) {
        const Eigen::Index r_start = bi * Br;
        const Eigen::Index br = std::min(Br, N - r_start);
        const Eigen::Index r_end = r_start + br;

        auto Q_block = Q.middleRows(r_start, br);
        auto dO_block = dO.middleRows(r_start, br);

        AttentionMatrixf dQ_acc = AttentionMatrixf::Zero(br, 64);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S_block(br, Bc);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dP_block(br, Bc);

        for (Eigen::Index bj = 0; bj < num_kv_blocks; ++bj) {
            const Eigen::Index c_start = bj * Bc;
            const Eigen::Index bc = std::min(Bc, N - c_start);
            const Eigen::Index c_end = c_start + bc;

            if (causal && c_start >= r_end) {
                break;
            }

            auto K_block = K.middleRows(c_start, bc);
            auto V_block = V.middleRows(c_start, bc);

            auto S_sub = S_block.leftCols(bc);
            auto dP_sub = dP_block.leftCols(bc);

            S_sub.noalias() = (Q_block * K_block.transpose()) * scale;

            if (causal && c_end > r_start) {
                for (Eigen::Index r = 0; r < br; ++r) {
                    const Eigen::Index gr = r_start + r;
                    for (Eigen::Index c = 0; c < bc; ++c) {
                        if (c_start + c > gr) {
                            S_sub(r, c) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }

            dP_sub.noalias() = dO_block * V_block.transpose();

            for (Eigen::Index r = 0; r < br; ++r) {
                const float lse = L(r_start + r);
                if (std::isfinite(lse)) {
                    auto p_row = (S_sub.row(r).array() - lse).exp();
                    S_sub.row(r).array() = p_row * (dP_sub.row(r).array() - D(r_start + r));
                } else {
                    S_sub.row(r).setZero();
                }
            }

            dQ_acc.noalias() += (S_sub * K_block) * scale;
        }

        grads.dQ.middleRows(r_start, br).noalias() = dQ_acc;
    }

    return grads;
}

// ============================================================================
// Locking FlashAttention-2 Backward (single K/V sweep, as in the FlashAttention-2 paper)
// ============================================================================

namespace detail {

/**
 * Test-and-test-and-set spinlock for the short dQ block updates below. Annotated for Cilksan,
 * so lock-protected accumulation is not reported as a race. Cache-line aligned so neighbouring
 * locks do not share a line.
 */
class alignas(64) BlockSpinLock {
public:
    // Cilksan only honours acquire/release annotations on locks registered with it
    BlockSpinLock() noexcept { __cilksan_register_lock_explicit(this); }
    ~BlockSpinLock() { __cilksan_unregister_lock_explicit(this); }
    BlockSpinLock(const BlockSpinLock&) = delete;
    BlockSpinLock& operator=(const BlockSpinLock&) = delete;

    void lock() noexcept {
        for (;;) {
            if (!locked_.exchange(true, std::memory_order_acquire)) {
                break;
            }
            while (locked_.load(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
        }
        __cilksan_acquire_lock(this);
    }

    void unlock() noexcept {
        __cilksan_release_lock(this);
        locked_.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> locked_{false};
};

} // namespace detail

/**
 * Locking Variant of FlashAttention-2 Backward:
 *
 * One parallel sweep over K/V blocks. Each task keeps dK_j and dV_j private and visits every
 * query block once, computing S, P, dP and dS a single time per tile (5 matrix products, against
 * 7 for the lock-free variant, which recomputes S and dP in a second sweep for dQ). Different
 * K/V tasks add into the same dQ_i rows, so each tile's dQ contribution is formed in a private
 * buffer and added to dQ_i under a per-query-block lock (the CPU analogue of the paper's atomic
 * adds). Each task starts at a different query block so tasks do not all queue on the same lock.
 *
 * The order of the dQ additions depends on scheduling, so dQ is not bitwise reproducible.
 */
inline AttentionGradients flash_attention_2_backward_locking_opencilk(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    const AttentionMatrixf& O,
    const AttentionMatrixf& dO,
    const Eigen::VectorXf& L,
    bool causal = false,
    Eigen::Index Br = 64,
    Eigen::Index Bc = 64
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));

    AttentionGradients grads{
        AttentionMatrixf(N, 64),
        AttentionMatrixf(N, 64),
        AttentionMatrixf(N, 64)
    };

    Eigen::VectorXf D(N);
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        D(i) = dO.row(i).dot(O.row(i));
        grads.dQ.row(i).setZero();
    }

    const Eigen::Index num_q_blocks = (N + Br - 1) / Br;
    const Eigen::Index num_kv_blocks = (N + Bc - 1) / Bc;
    std::vector<detail::BlockSpinLock> dq_locks(num_q_blocks);

    cilk_for (Eigen::Index bj = 0; bj < num_kv_blocks; ++bj) {
        const Eigen::Index c_start = bj * Bc;
        const Eigen::Index bc = std::min(Bc, N - c_start);
        const Eigen::Index c_end = c_start + bc;

        auto K_block = K.middleRows(c_start, bc);
        auto V_block = V.middleRows(c_start, bc);

        AttentionMatrixf dK_acc = AttentionMatrixf::Zero(bc, 64);
        AttentionMatrixf dV_acc = AttentionMatrixf::Zero(bc, 64);

        // Task-local tile buffers: P is formed in place over S, and dS in place over dP
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S_block(Br, bc);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dP_block(Br, bc);
        AttentionMatrixf dQ_tile(Br, 64);

        for (Eigen::Index t = 0; t < num_q_blocks; ++t) {
            const Eigen::Index bi = (bj + t) % num_q_blocks; // Staggered start avoids a lock convoy
            const Eigen::Index r_start = bi * Br;
            const Eigen::Index br = std::min(Br, N - r_start);
            const Eigen::Index r_end = r_start + br;

            // Causal block skipping: every key in this block is after every query in block bi
            if (causal && c_start >= r_end) {
                continue;
            }

            auto Q_block = Q.middleRows(r_start, br);
            auto dO_block = dO.middleRows(r_start, br);

            auto P_sub = S_block.topRows(br);
            auto dS_sub = dP_block.topRows(br);

            // 1. S = Q_i K_j^T * scale, then P = exp(S - L_i) in place
            P_sub.noalias() = (Q_block * K_block.transpose()) * scale;
            if (causal && c_end > r_start) {
                for (Eigen::Index r = 0; r < br; ++r) {
                    const Eigen::Index gr = r_start + r;
                    for (Eigen::Index c = 0; c < bc; ++c) {
                        if (c_start + c > gr) {
                            P_sub(r, c) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
            for (Eigen::Index r = 0; r < br; ++r) {
                const float lse = L(r_start + r);
                if (std::isfinite(lse)) {
                    P_sub.row(r) = (P_sub.row(r).array() - lse).exp();
                } else {
                    P_sub.row(r).setZero();
                }
            }

            // 2. dV_j += P^T dO_i
            dV_acc.noalias() += P_sub.transpose() * dO_block;

            // 3. dP = dO_i V_j^T, then dS = P * (dP - D_i) in place
            dS_sub.noalias() = dO_block * V_block.transpose();
            for (Eigen::Index r = 0; r < br; ++r) {
                dS_sub.row(r).array() = P_sub.row(r).array() * (dS_sub.row(r).array() - D(r_start + r));
            }

            // 4. dK_j += dS^T Q_i * scale
            dK_acc.noalias() += (dS_sub.transpose() * Q_block) * scale;

            // 5. dQ_i += dS K_j * scale: private product, shared update under block bi's lock
            auto dQ_sub = dQ_tile.topRows(br);
            dQ_sub.noalias() = (dS_sub * K_block) * scale;
            dq_locks[bi].lock();
            grads.dQ.middleRows(r_start, br) += dQ_sub;
            dq_locks[bi].unlock();
        }

        grads.dK.middleRows(c_start, bc).noalias() = dK_acc;
        grads.dV.middleRows(c_start, bc).noalias() = dV_acc;
    }

    return grads;
}
