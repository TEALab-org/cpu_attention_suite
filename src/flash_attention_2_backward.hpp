// ============================================================================
// 3. Strictly Lock-Free FlashAttention-2 Backward (For Cilkscale / Cilksan)
// ============================================================================

/**
 * Lock-Free Series-Parallel DAG Variant of FlashAttention-2 Backward:
 *
 * Eliminates all mutexes/atomics by splitting the Q-accumulation and KV-accumulation
 * into two lock-free parallel loops. Ideal when running under OpenCilk's Cilksan
 * race detector or Cilkscale work-span profiler.
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
