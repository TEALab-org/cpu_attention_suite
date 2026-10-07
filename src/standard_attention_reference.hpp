/**
 * Standard Scaled Dot-Product Attention Reference:
 *   S = (Q * K^T) / sqrt(d)
 *   P = softmax(S, row-wise)
 *   O = P * V
 *
 * Runs as a high-precision FP32 ground truth for numerical parity testing.
 */
inline AttentionMatrixf standard_attention_reference(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    bool causal = false
) {
    const Eigen::Index N = Q.rows();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Q.cols()));

    // Allocate full N x N affinity score matrix.
    // Stored RowMajor so row-wise softmax scans contiguous memory lines.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S(N, N);

    // Compute S = (Q * K^T) * scale
    // Q is (N, 64), K^T is (64, N) -> S is (N, N)
    S.noalias() = (Q * K.transpose()) * scale;

    // Numerically stable row-wise Softmax: P_ij = exp(S_ij - m_i) / sum_j(exp(S_ij - m_i))
    for (Eigen::Index i = 0; i < N; ++i) {
        auto row = S.row(i);

        if (causal && i + 1 < N) {
            // Mask future tokens (j > i) with -inf
            row.tail(N - 1 - i).setConstant(-std::numeric_limits<float>::infinity());
        }

        // 1. Row maximum for stability against overflow
        const float m_i = row.maxCoeff();

        // 2. Element-wise exponentiation
        row = (row.array() - m_i).exp();

        // 3. Normalization factor
        const float l_i = row.sum();

        // 4. In-place probability normalization
        if (l_i > 0.0f) {
            row.array() /= l_i;
        } else {
            row.setZero();
        }
    }

    // Allocate output matrix O (N, 64)
    AttentionMatrixf O(N, 64);

    // Compute O = P * V
    // P (stored in S) is (N, N) RowMajor, V is (N, 64) RowMajor -> O is (N, 64) RowMajor
    O.noalias() = S * V;

    return O;
}
