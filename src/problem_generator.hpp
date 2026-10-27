#pragma once

#include "attention_utilities.hpp"
#include <cilk/cilk.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace detail {

/// SplitMix64 step: small, fast, statistically solid 64-bit generator.
inline uint64_t splitmix64(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// Fills matrix rows with uniform [-1, 1) values. Each row has its own stream derived from
/// (seed, stream_id, row), so the result is independent of the worker count and schedule.
inline void fill_uniform(AttentionMatrixf& M, uint64_t seed, uint64_t stream_id) {
    cilk_for (Eigen::Index r = 0; r < M.rows(); ++r) {
        uint64_t state = seed ^ (stream_id << 56) ^ (static_cast<uint64_t>(r) * 0xD1B54A32D192ED03ull);
        splitmix64(state); // Decorrelate neighbouring row seeds
        for (Eigen::Index c = 0; c < M.cols(); ++c) {
            // Top 24 bits -> float in [0, 1) exactly, then map to [-1, 1)
            const float u = static_cast<float>(splitmix64(state) >> 40) * (1.0f / 16777216.0f);
            M(r, c) = 2.0f * u - 1.0f;
        }
    }
}

} // namespace detail

/**
 * Synthetic (N, 64) Q, K, V (and optionally dO) problem with entries uniform in [-1, 1), matching the
 * distribution of python_scripts/generate_problem.py (but not its exact values).
 * Avoids storing multi-GB .npy inputs for large benchmark sizes.
 */
inline AttentionProblem generate_random_problem(Eigen::Index N, uint64_t seed, bool with_dO = false) {
    AttentionProblem problem;
    problem.Q.resize(N, 64);
    problem.K.resize(N, 64);
    problem.V.resize(N, 64);
    detail::fill_uniform(problem.Q, seed, 0);
    detail::fill_uniform(problem.K, seed, 1);
    detail::fill_uniform(problem.V, seed, 2);
    if (with_dO) {
        problem.dO.resize(N, 64);
        detail::fill_uniform(problem.dO, seed, 3);
    }
    return problem;
}

/**
 * Sets a fraction of rows exactly to zero in aligned runs of `run` rows: the n / run runs are
 * shuffled with a seeded generator and the first round(fraction * runs) are zeroed in every given
 * matrix (several matrices: jointly zero rows, as for K and V). Clustered runs let the interval
 * tests c(I) = 0 fire; run = 1 disperses zero rows. Returns the number of rows zeroed.
 */
inline Eigen::Index zero_row_runs(std::initializer_list<AttentionMatrixf*> matrices, double fraction,
                                  Eigen::Index run, uint64_t seed) {
    if (fraction <= 0.0) {
        return 0;
    }
    const Eigen::Index n = (*matrices.begin())->rows();
    run = std::max<Eigen::Index>(1, std::min(run, n));
    const Eigen::Index runs = (n + run - 1) / run;
    std::vector<Eigen::Index> order(runs);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    const auto chosen = static_cast<Eigen::Index>(std::llround(std::min(1.0, fraction) * static_cast<double>(runs)));
    Eigen::Index zeroed = 0;
    for (Eigen::Index k = 0; k < chosen; ++k) {
        const Eigen::Index start = order[k] * run;
        const Eigen::Index len = std::min(run, n - start);
        for (AttentionMatrixf* m : matrices) {
            m->middleRows(start, len).setZero();
        }
        zeroed += len;
    }
    return zeroed;
}
