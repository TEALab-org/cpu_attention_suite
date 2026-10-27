#pragma once

// Numerical row activity (Section 3.3.1.2): a_i = [row i is not exactly zero], with prefix counts
// so that c(I) = |{active rows in I}| is an O(1) query. Discovery scans every row (Theta(Nd))
// and is timed as part of the sparse solvers, as the paper charges it.

#include "dense_accumulate.hpp" // IndexRange
#include <cilk/cilk.h>

#include <algorithm>
#include <cstdint>
#include <vector>

class RowActivity {
public:
    RowActivity() = default;

    /// Every row active: counts equal interval lengths (the shortcut is disabled)
    static RowActivity all_active(Eigen::Index n) {
        RowActivity activity;
        activity.n_ = n;
        activity.all_active_ = true;
        return activity;
    }

    /// Active where any row of the given matrices is nonzero (several matrices: jointly zero rows)
    static RowActivity from_rows(std::initializer_list<const AttentionMatrixf*> matrices) {
        const Eigen::Index n = (*matrices.begin())->rows();
        RowActivity activity;
        activity.n_ = n;
        std::vector<int64_t> flags(n);
        cilk_for (Eigen::Index i = 0; i < n; ++i) {
            bool active = false;
            for (const AttentionMatrixf* m : matrices) {
                active = active || (m->row(i).array() != 0.0f).any();
            }
            flags[i] = active ? 1 : 0;
        }
        activity.prefix_ = parallel_prefix(flags);
        return activity;
    }

    [[nodiscard]] Eigen::Index count(IndexRange I) const noexcept {
        if (all_active_) {
            return I.len;
        }
        return static_cast<Eigen::Index>(prefix_[I.start + I.len] - prefix_[I.start]);
    }

    [[nodiscard]] bool active(Eigen::Index i) const noexcept { return count({i, 1}) == 1; }
    [[nodiscard]] Eigen::Index total() const noexcept { return count({0, n_}); }

private:
    Eigen::Index n_ = 0;
    bool all_active_ = false;
    std::vector<int64_t> prefix_; // prefix_[i] = active rows in [0, i)

    /// Exclusive prefix sum (length n + 1) as a blocked two-pass parallel scan
    static std::vector<int64_t> parallel_prefix(const std::vector<int64_t>& flags) {
        const auto n = static_cast<Eigen::Index>(flags.size());
        constexpr Eigen::Index kBlock = 4096;
        const Eigen::Index blocks = (n + kBlock - 1) / kBlock;
        std::vector<int64_t> block_sums(blocks + 1, 0);
        cilk_for (Eigen::Index blk = 0; blk < blocks; ++blk) {
            int64_t sum = 0;
            for (Eigen::Index i = blk * kBlock; i < std::min(n, (blk + 1) * kBlock); ++i) {
                sum += flags[i];
            }
            block_sums[blk + 1] = sum;
        }
        for (Eigen::Index blk = 0; blk < blocks; ++blk) {
            block_sums[blk + 1] += block_sums[blk];
        }
        std::vector<int64_t> prefix(n + 1, 0);
        cilk_for (Eigen::Index blk = 0; blk < blocks; ++blk) {
            int64_t running = block_sums[blk];
            for (Eigen::Index i = blk * kBlock; i < std::min(n, (blk + 1) * kBlock); ++i) {
                prefix[i] = running;
                running += flags[i];
            }
        }
        prefix[n] = block_sums[blocks];
        return prefix;
    }
};
