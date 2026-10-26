#pragma once

#include "attention_utilities.hpp"
#include <cilk/cilk.h>
#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================================
// 1. OpenCilk Standard Attention Backward Pass (O(N^2) Memory)
// ============================================================================

/**
 * Standard Attention Backward Pass (OpenCilk Parallelized):
 *   D_i  = dO_i . O_i
 *   dV   = P^T * dO
 *   dP   = dO * V^T
 *   dS_i = P_i \odot (dP_i - D_i)
 *   dQ   = (dS * K) / sqrt(d)
 *   dK   = (dS^T * Q) / sqrt(d)
 *
 * Materializes full N x N matrices P and dS in global memory to serve as
 * both a numerical ground truth and an O(N^2) memory-bandwidth baseline.
 */
inline AttentionGradients standard_attention_backward_opencilk(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    const AttentionMatrixf& O,
    const AttentionMatrixf& dO,
    bool causal = false,
    Eigen::Index Bc = 64 // Column block size for cache-friendly P^T and dS^T GEMMs
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));

    // Allocate full N x N probability and score-gradient matrices in RowMajor order
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(N, N);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dS(N, N);

    AttentionGradients grads{
        AttentionMatrixf(N, 64),
        AttentionMatrixf(N, 64),
        AttentionMatrixf(N, 64)
    };

    // Phase 1: Recompute P = softmax(Q * K^T / sqrt(d)) in parallel over rows
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        auto p_row = P.row(i);
        p_row.noalias() = (Q.row(i) * K.transpose()) * scale;

        if (causal && i + 1 < N) {
            p_row.tail(N - 1 - i).setConstant(-std::numeric_limits<float>::infinity());
        }

        const float m_i = p_row.maxCoeff();
        p_row = (p_row.array() - m_i).exp();
        const float l_i = p_row.sum();

        if (l_i > 0.0f) {
            p_row.array() /= l_i;
        } else {
            p_row.setZero();
        }
    }

    // Phase 2: Compute dS_i = P_i \odot (dO_i * V^T - D_i) and dQ_i = scale * (dS_i * K)
    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        // Identity: sum_j(dP_ij * P_ij) == dO_i . O_i
        const float D_i = dO.row(i).dot(O.row(i));

        auto ds_row = dS.row(i);
        // Temporarily store dP_i = dO_i * V^T in ds_row
        ds_row.noalias() = dO.row(i) * V.transpose();

        // Element-wise softmax backward: dS_ij = P_ij * (dP_ij - D_i)
        ds_row.array() = P.row(i).array() * (ds_row.array() - D_i);

        // Row gradient dQ_i = scale * (dS_i * K)
        grads.dQ.row(i).noalias() = (ds_row * K) * scale;
    }

    // Phase 3: Compute dV = P^T * dO and dK = scale * (dS^T * Q)
    // Tiled across column blocks of P and dS so RowMajor transposes execute as Level-3 BLAS GEMMs
    const Eigen::Index num_kv_blocks = (N + Bc - 1) / Bc;
    cilk_for (Eigen::Index bj = 0; bj < num_kv_blocks; ++bj) {
        const Eigen::Index c_start = bj * Bc;
        const Eigen::Index bc = std::min(Bc, N - c_start);

        grads.dV.middleRows(c_start, bc).noalias() =
            P.middleCols(c_start, bc).transpose() * dO;

        grads.dK.middleRows(c_start, bc).noalias() =
            (dS.middleCols(c_start, bc).transpose() * Q) * scale;
    }

    return grads;
}
