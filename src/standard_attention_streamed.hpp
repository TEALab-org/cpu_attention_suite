#pragma once

#include "attention_utilities.hpp"
#include <cmath>
#include <limits>
#include <cilk/cilk.h>

/**
 * Fused Row-Streamed Attention:
 * Computes exact standard attention without materializing S in global DRAM.
 * Peak working memory drops from O(N^2) to O(num_workers * N).
 */
inline AttentionMatrixf standard_attention_streamed_opencilk(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    bool causal = false
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));
    AttentionMatrixf O(N, 64);

    cilk_for (Eigen::Index i = 0; i < N; ++i) {
        // Allocate a private row buffer on the worker thread's stack/heap
        Eigen::RowVectorXf s_i(N);

        // 1. Q_i * K^T
        s_i.noalias() = (Q.row(i) * K.transpose()) * scale;

        // 2. Causal masking + Softmax in L1/L2 cache
        if (causal && i + 1 < N) {
            s_i.tail(N - 1 - i).setConstant(-std::numeric_limits<float>::infinity());
        }
        const float m_i = s_i.maxCoeff();
        s_i = (s_i.array() - m_i).exp();
        const float l_i = s_i.sum();

        if (l_i > 0.0f) {
            s_i.array() /= l_i;
        } else {
            s_i.setZero();
        }

        // 3. O_i = P_i * V
        O.row(i).noalias() = s_i * V;
    }

    return O;
}
