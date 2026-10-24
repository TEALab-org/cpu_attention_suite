#pragma once

#include "attention_utilities.hpp" // Provides AttentionMatrixf and AttentionProblem
#include <cilk/cilk.h>
#include <cmath>
#include <limits>
#include <algorithm>

/**
 * FlashAttention-2 Forward Pass (CPU + OpenCilk):
 *
 * Algorithm Key Highlights:
 * 1. Outer loop parallelized over Query blocks Q_i: eliminates write hazards
 *    and thread synchronization on the output matrix O.
 * 2. Unnormalized accumulator O_acc: defers division by the row sum l_i
 *    until the full sequence of K, V blocks has been processed.
 * 3. Tiled L1/L2 memory footprint: the working buffer S (reused in place as P) is a small
 *    (Br x Bc) block that stays in CPU cache.
 */
inline AttentionMatrixf flash_attention_2(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    bool causal = false,
    Eigen::Index Br = 64, // Block size along sequence rows (queries)
    Eigen::Index Bc = 64  // Block size along sequence cols (keys/values)
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));

    AttentionMatrixf O(N, 64);

    const Eigen::Index num_q_blocks = (N + Br - 1) / Br;
    const Eigen::Index num_kv_blocks = (N + Bc - 1) / Bc;

    // FlashAttention-2 parallelizes over Q blocks: each task owns an independent
    // horizontal slice O[r_start : r_end, :] without atomic contention.
    cilk_for (Eigen::Index bi = 0; bi < num_q_blocks; ++bi) {
        const Eigen::Index r_start = bi * Br;
        const Eigen::Index br = std::min(Br, N - r_start);
        const Eigen::Index r_end = r_start + br;

        // Slice Q block: (br, 64)
        auto Q_block = Q.middleRows(r_start, br);

        // Per-row online softmax state buffers
        Eigen::VectorXf m = Eigen::VectorXf::Constant(br, -std::numeric_limits<float>::infinity());
        Eigen::VectorXf l = Eigen::VectorXf::Zero(br);

        // FlashAttention-2 unnormalized output accumulator \tilde{O}_i, accumulated directly
        // in this task's (disjoint) slice of O
        auto O_acc = O.middleRows(r_start, br);
        O_acc.setZero();

        // Thread-local tile buffer, allocated once per query block. S is exponentiated
        // in place into P = exp(S - m_new), so no separate P buffer is needed.
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S(br, Bc);

        // Inner loop over Key and Value blocks
        for (Eigen::Index bj = 0; bj < num_kv_blocks; ++bj) {
            const Eigen::Index c_start = bj * Bc;
            const Eigen::Index bc = std::min(Bc, N - c_start);
            const Eigen::Index c_end = c_start + bc;

            // Causal Block Skipping:
            // If the key block starts beyond the query block's end, all its cells are masked.
            // Because bj is monotonic, all subsequent blocks are also causal masked.
            if (causal && c_start >= r_end) {
                break;
            }

            auto K_block = K.middleRows(c_start, bc);
            auto V_block = V.middleRows(c_start, bc);
            auto S_tile = S.leftCols(bc);

            // 1. Tiled affinity GEMM: S_ij = (Q_i * K_j^T) * scale
            // Result size: (br, bc)
            S_tile.noalias() = (Q_block * K_block.transpose()) * scale;

            // 2. Intra-block causal diagonal masking
            if (causal && c_end > r_start) {
                for (Eigen::Index r = 0; r < br; ++r) {
                    const Eigen::Index gr = r_start + r;
                    for (Eigen::Index c = 0; c < bc; ++c) {
                        const Eigen::Index gc = c_start + c;
                        if (gc > gr) {
                            S_tile(r, c) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }

            // 3. Online softmax update, one row at a time:
            //   m_new = max(m, rowmax(S)),  \alpha = exp(m - m_new),  P = exp(S - m_new)
            //   l = \alpha * l + rowsum(P),  O_acc = diag(\alpha) * O_acc
            for (Eigen::Index r = 0; r < br; ++r) {
                auto s_row = S_tile.row(r);
                const float m_new = std::max(m(r), s_row.maxCoeff());

                if (m_new == -std::numeric_limits<float>::infinity()) {
                    // Row fully masked so far: contributes nothing (avoids exp(-inf - -inf) = NaN)
                    s_row.setZero();
                    continue;
                }

                const float alpha = (m(r) == -std::numeric_limits<float>::infinity())
                    ? 0.0f
                    : std::exp(m(r) - m_new);

                s_row = (s_row.array() - m_new).exp();
                l(r) = alpha * l(r) + s_row.sum();
                O_acc.row(r) *= alpha;
                m(r) = m_new;
            }

            // 4. Update unnormalized output accumulator: O_acc += P * V_block
            O_acc.noalias() += S_tile * V_block;
        }

        // 5. Final FlashAttention-2 normalization:
        // O_i = diag(1 / l_i) * O_acc
        for (Eigen::Index r = 0; r < br; ++r) {
            if (l(r) > 0.0f) {
                O_acc.row(r) /= l(r);
            } else {
                O_acc.row(r).setZero();
            }
        }
    }

    return O;
}
