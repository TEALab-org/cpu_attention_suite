#pragma once

// Parallel cache-oblivious dense backward passes:
//   ParallelDenseBackward  (Algorithms 7-9): 4-way recursion with private gradient summaries
//   ParallelOwnedBackward  (Algorithms 7, 10): two traversals with disjoint output ownership
//   Hybrid                 : summary recursion while p splits 4 ways, then owned traversals

#include "dense_backward.hpp"
#include <cilk/cilk.h>
#include <cmath>

namespace detail {

/**
 * ParallelBackwardSummary(I, J, p, \Gamma) (Algorithm 8): spawns 4 quadrant subtasks with private
 * gradient summaries until the budget p reaches 1, then accumulates serially and merges upward.
 */
inline void parallel_backward_summary(
    const BackwardInputs& in,
    IndexRange I,
    IndexRange J,
    int p,
    GradientSummary& gamma,
    Eigen::Index b
) {
    // Base Case: p == 1 (or the subproblem has already reached leaf granularity)
    if (p <= 1 || (I.len <= b && J.len <= b)) {
        reset_gradients(gamma, I, J);
        BackwardLeafWorkspace ws(b);
        backward_accumulate(in, I, J, gamma, GradientMode::All, b, ws);
        return;
    }

    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    const IndexRange I1{I.start, i_mid};
    const IndexRange I2{I.start + i_mid, I.len - i_mid};
    const IndexRange J1{J.start, j_mid};
    const IndexRange J2{J.start + j_mid, J.len - j_mid};

    // \Gamma_{ab'} <- AllocateGradients(I_a, J_{b'}, d); freed via RAII after the merge
    GradientSummary gamma_11(I1, J1);
    GradientSummary gamma_12(I1, J2);
    GradientSummary gamma_21(I2, J1);
    GradientSummary gamma_22(I2, J2);

    const int child_p = p / 4;

    cilk_spawn parallel_backward_summary(in, I1, J1, child_p, gamma_11, b);
    cilk_spawn parallel_backward_summary(in, I1, J2, child_p, gamma_12, b);
    cilk_spawn parallel_backward_summary(in, I2, J1, child_p, gamma_21, b);
               parallel_backward_summary(in, I2, J2, child_p, gamma_22, b);
    cilk_sync;

    parallel_gradient_merge(I, J, gamma, gamma_11, gamma_12, gamma_21, gamma_22);
}

/**
 * OwnedBackward(I, J, mode, \Gamma) (Algorithm 10). Mode Q writes only \Delta Q and runs the two
 * query halves in parallel against each key half in turn; mode KV writes only (\Delta K, \Delta V)
 * and runs the two key halves in parallel against each query half in turn. Concurrent tasks
 * therefore always write disjoint rows of the shared \Gamma.
 *
 * Once the owned interval (I in mode Q, J in mode KV) is at most grain, the rest of the (I, J)
 * block runs as a serial BackwardAccumulate with one task-local workspace, as in QueryParallel.
 */
inline void owned_backward(
    const BackwardInputs& in,
    IndexRange I,
    IndexRange J,
    GradientMode mode,
    GradientSummary& gamma,
    Eigen::Index b,
    Eigen::Index grain
) {
    if (I.len <= 0 || J.len <= 0) {
        return;
    }

    const Eigen::Index owned_len = (mode == GradientMode::Q) ? I.len : J.len;
    if (owned_len <= grain || (I.len <= b && J.len <= b)) {
        BackwardLeafWorkspace ws(b);
        backward_accumulate(in, I, J, gamma, mode, b, ws);
        return;
    }

    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    const IndexRange I1{I.start, i_mid};
    const IndexRange I2{I.start + i_mid, I.len - i_mid};
    const IndexRange J1{J.start, j_mid};
    const IndexRange J2{J.start + j_mid, J.len - j_mid};

    if (mode == GradientMode::Q) {
        cilk_spawn owned_backward(in, I1, J1, mode, gamma, b, grain);
                   owned_backward(in, I2, J1, mode, gamma, b, grain);
        cilk_sync;
        cilk_spawn owned_backward(in, I1, J2, mode, gamma, b, grain);
                   owned_backward(in, I2, J2, mode, gamma, b, grain);
        cilk_sync;
    } else {
        cilk_spawn owned_backward(in, I1, J1, mode, gamma, b, grain);
                   owned_backward(in, I1, J2, mode, gamma, b, grain);
        cilk_sync;
        cilk_spawn owned_backward(in, I2, J1, mode, gamma, b, grain);
                   owned_backward(in, I2, J2, mode, gamma, b, grain);
        cilk_sync;
    }
}

/**
 * HybridBackwardSummary(I, J, p, \Gamma): ParallelBackwardSummary's 4-way recursion while p
 * divides evenly by 4, then both owned traversals on each remaining subproblem with a fresh
 * \Gamma, mirroring the forward hybrid. Any p >= 1 works.
 */
inline void hybrid_backward_summary(
    const BackwardInputs& in,
    IndexRange I,
    IndexRange J,
    int p,
    GradientSummary& gamma,
    Eigen::Index b,
    Eigen::Index grain
) {
    // Switch: p no longer splits evenly 4 ways (or subproblem has already reached leaf granularity)
    if (p % 4 != 0 || (I.len <= b && J.len <= b)) {
        reset_gradients(gamma, I, J);
        owned_backward(in, I, J, GradientMode::Q, gamma, b, grain);
        owned_backward(in, I, J, GradientMode::KV, gamma, b, grain);
        return;
    }

    const Eigen::Index i_mid = I.len / 2;
    const Eigen::Index j_mid = J.len / 2;

    const IndexRange I1{I.start, i_mid};
    const IndexRange I2{I.start + i_mid, I.len - i_mid};
    const IndexRange J1{J.start, j_mid};
    const IndexRange J2{J.start + j_mid, J.len - j_mid};

    GradientSummary gamma_11(I1, J1);
    GradientSummary gamma_12(I1, J2);
    GradientSummary gamma_21(I2, J1);
    GradientSummary gamma_22(I2, J2);

    const int child_p = p / 4;

    cilk_spawn hybrid_backward_summary(in, I1, J1, child_p, gamma_11, b, grain);
    cilk_spawn hybrid_backward_summary(in, I1, J2, child_p, gamma_12, b, grain);
    cilk_spawn hybrid_backward_summary(in, I2, J1, child_p, gamma_21, b, grain);
               hybrid_backward_summary(in, I2, J2, child_p, gamma_22, b, grain);
    cilk_sync;

    parallel_gradient_merge(I, J, gamma, gamma_11, gamma_12, gamma_21, gamma_22);
}

} // namespace detail

/**
 * ParallelDenseBackward(Q, K, V, O, dO, L, p) (Algorithm 7).
 * @param p Processor budget (a power of 4, as in the forward ParallelSummary)
 * @param b Base case block threshold
 */
inline AttentionGradients parallel_dense_backward(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    const AttentionMatrixf& O,
    const AttentionMatrixf& dO,
    const Eigen::VectorXf& L,
    int p,
    Eigen::Index b = 64
) {
    const Eigen::Index N = Q.rows();
    const Eigen::VectorXf D = detail::output_gradient_dots(dO, O);
    const BackwardInputs in{Q, K, V, dO, L, D, 1.0f / std::sqrt(static_cast<float>(Q.cols()))};
    const IndexRange full_range{0, N};

    // \Gamma needs no initialization: every entry is written by a base case or a merge
    GradientSummary gamma(full_range, full_range);
    detail::parallel_backward_summary(in, full_range, full_range, p, gamma, b);
    return detail::take_gradients(gamma);
}

/**
 * ParallelOwnedBackward(Q, K, V, O, dO, L) (Algorithm 7): zero \Gamma, then the Q-mode
 * traversal followed by the KV-mode traversal.
 * @param b     Base case block threshold
 * @param grain Owned-interval size below which each traversal continues serially
 */
inline AttentionGradients parallel_owned_backward(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    const AttentionMatrixf& O,
    const AttentionMatrixf& dO,
    const Eigen::VectorXf& L,
    Eigen::Index b = 64,
    Eigen::Index grain = 256
) {
    const Eigen::Index N = Q.rows();
    const Eigen::VectorXf D = detail::output_gradient_dots(dO, O);
    const BackwardInputs in{Q, K, V, dO, L, D, 1.0f / std::sqrt(static_cast<float>(Q.cols()))};
    const IndexRange full_range{0, N};

    GradientSummary gamma(full_range, full_range);
    detail::reset_gradients(gamma, full_range, full_range);
    detail::owned_backward(in, full_range, full_range, GradientMode::Q, gamma, b, grain);
    detail::owned_backward(in, full_range, full_range, GradientMode::KV, gamma, b, grain);
    return detail::take_gradients(gamma);
}

/**
 * Hybrid ParallelDenseBackward / ParallelOwnedBackward driver.
 * @param p     Processor budget (any p >= 1)
 * @param b     Base case block threshold
 * @param grain Owned-interval size below which the owned traversals continue serially
 */
inline AttentionGradients hybrid_backward(
    const AttentionMatrixf& Q,
    const AttentionMatrixf& K,
    const AttentionMatrixf& V,
    const AttentionMatrixf& O,
    const AttentionMatrixf& dO,
    const Eigen::VectorXf& L,
    int p,
    Eigen::Index b = 64,
    Eigen::Index grain = 256
) {
    const Eigen::Index N = Q.rows();
    const Eigen::VectorXf D = detail::output_gradient_dots(dO, O);
    const BackwardInputs in{Q, K, V, dO, L, D, 1.0f / std::sqrt(static_cast<float>(Q.cols()))};
    const IndexRange full_range{0, N};

    GradientSummary gamma(full_range, full_range);
    detail::hybrid_backward_summary(in, full_range, full_range, p, gamma, b, grain);
    return detail::take_gradients(gamma);
}
