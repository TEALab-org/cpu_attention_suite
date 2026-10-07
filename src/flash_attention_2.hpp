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
 * 3. Tiled L1/L2 memory footprint: working buffers S and P are small
 *    (Br x Bc) blocks that stay in CPU cache lines.
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

        // FlashAttention-2 unnormalized output accumulator: \tilde{O}_i = P_i * V
        Eigen::Matrix<float, Eigen::Dynamic, 64, Eigen::RowMajor> O_acc = 
            Eigen::Matrix<float, Eigen::Dynamic, 64, Eigen::RowMajor>::Zero(br, 64);

        // Preallocated thread-local tile buffers
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S(br, Bc);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(br, Bc);

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

            // 1. Tiled affinity GEMM: S_ij = (Q_i * K_j^T) * scale
            // Result size: (br, bc)
            S.leftCols(bc).noalias() = (Q_block * K_block.transpose()) * scale;

            // 2. Intra-block causal diagonal masking
            if (causal && c_end > r_start) {
                for (Eigen::Index r = 0; r < br; ++r) {
                    const Eigen::Index gr = r_start + r;
                    for (Eigen::Index c = 0; c < bc; ++c) {
                        const Eigen::Index gc = c_start + c;
                        if (gc > gr) {
                            S(r, c) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }

            // 3. Online Softmax update
            // Row-wise block maximum \tilde{m}_{ij}
            Eigen::VectorXf m_tilde = S.leftCols(bc).rowwise().maxCoeff();
            Eigen::VectorXf m_new = m.cwiseMax(m_tilde);

            // Rescaling factor \alpha = exp(m - m_new)
            Eigen::VectorXf alpha(br);
            for (Eigen::Index r = 0; r < br; ++r) {
                alpha(r) = (m(r) == -std::numeric_limits<float>::infinity())
                    ? 0.0f
                    : std::exp(m(r) - m_new(r));
            }

            // Unnormalized exponentiated probabilities: P = exp(S - m_new)
            for (Eigen::Index r = 0; r < br; ++r) {
                P.row(r).head(bc) = (S.row(r).head(bc).array() - m_new(r)).exp();
            }

            // Row-wise sum of current block exponentials
            Eigen::VectorXf p_sum = P.leftCols(bc).rowwise().sum();

            // Running sum update: l_new = alpha * l + rowsum(P)
            l = alpha.cwiseProduct(l) + p_sum;

            // 4. Update unnormalized output accumulator:
            // O_acc = diag(alpha) * O_acc + P * V_block
            for (Eigen::Index r = 0; r < br; ++r) {
                O_acc.row(r) *= alpha(r);
            }
            O_acc.noalias() += P.leftCols(bc) * V_block;

            m = m_new;
        }

        // 5. Final FlashAttention-2 normalization:
        // O_i = diag(1 / l_i) * O_acc
        for (Eigen::Index r = 0; r < br; ++r) {
            if (l(r) > 0.0f) {
                O.row(r_start + r).noalias() = O_acc.row(r) / l(r);
            } else {
                O.row(r_start + r).setZero();
            }
        }
    }

    return O;
}
