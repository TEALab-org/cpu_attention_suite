#pragma once

// Structural attention mask P supplied as a compressed quadtree aligned with the recursive
// partition of the score matrix (Section 3.3.1.1). Each stored node holds its allowed-pair count
// c(I, J); full and empty blocks are not stored (a parent's child slot holds kFull or kEmpty, and
// descendants of a full block are full by implication); a partial b x b leaf stores its allowed
// entries as a row-major bitmap. Construction is setup work and is not timed.

#include "dense_accumulate.hpp" // IndexRange
#include "problem_generator.hpp" // detail::splitmix64
#include <cilk/cilk.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/// Handle to a mask block: a stored node index (>= 0), or one of the implicit kinds below
using MaskHandle = int32_t;
constexpr MaskHandle kMaskFull = -1;
constexpr MaskHandle kMaskEmpty = -2;

class MaskQuadtree {
public:
    struct Node {
        int64_t count = 0;                         // c(I, J)
        MaskHandle child[4] = {kMaskEmpty, kMaskEmpty, kMaskEmpty, kMaskEmpty}; // (I1,J1),(I1,J2),(I2,J1),(I2,J2)
        int64_t bits = -1;                         // Partial leaf: word offset of its |I| x |J| bitmap
    };

    Eigen::Index n = 0;   // Sequence length (the mask is n x n)
    Eigen::Index b = 0;   // Leaf side; must equal the solver's base case size
    MaskHandle root = kMaskFull;
    std::vector<Node> nodes;
    std::vector<uint64_t> words;

    /// A full mask (no structural sparsity)
    static MaskQuadtree full(Eigen::Index n, Eigen::Index b) {
        MaskQuadtree tree;
        tree.n = n;
        tree.b = b;
        tree.root = kMaskFull;
        return tree;
    }

    [[nodiscard]] int64_t count(MaskHandle h, IndexRange I, IndexRange J) const noexcept {
        if (h == kMaskFull) {
            return static_cast<int64_t>(I.len) * static_cast<int64_t>(J.len);
        }
        if (h == kMaskEmpty) {
            return 0;
        }
        return nodes[h].count;
    }

    /// Child q of a block: 0 = (I1, J1), 1 = (I1, J2), 2 = (I2, J1), 3 = (I2, J2)
    [[nodiscard]] MaskHandle child(MaskHandle h, int q) const noexcept {
        return (h < 0) ? h : nodes[h].child[q];
    }

    /// Bitmap of a stored partial leaf (row-major over its |I| x |J| entries)
    [[nodiscard]] const uint64_t* leaf_bits(MaskHandle h) const noexcept {
        return words.data() + nodes[h].bits;
    }

    [[nodiscard]] static bool bit(const uint64_t* bits, Eigen::Index index) noexcept {
        return (bits[index >> 6] >> (index & 63)) & 1u;
    }

    [[nodiscard]] int64_t nnz() const noexcept { return count(root, {0, n}, {0, n}); }

    /// Expands to a dense row-major n x n 0/1 mask (tests and inspection at small n only)
    [[nodiscard]] std::vector<uint8_t> to_dense() const {
        std::vector<uint8_t> dense(static_cast<size_t>(n * n), 0);
        expand(root, {0, n}, {0, n}, dense);
        return dense;
    }

private:
    void expand(MaskHandle h, IndexRange I, IndexRange J, std::vector<uint8_t>& dense) const {
        if (h == kMaskEmpty || I.len <= 0 || J.len <= 0) {
            return;
        }
        if (h == kMaskFull) {
            for (Eigen::Index r = 0; r < I.len; ++r) {
                std::fill_n(dense.begin() + (I.start + r) * n + J.start, J.len, uint8_t{1});
            }
            return;
        }
        if (I.len <= b && J.len <= b) {
            const uint64_t* bits = leaf_bits(h);
            for (Eigen::Index r = 0; r < I.len; ++r) {
                for (Eigen::Index c = 0; c < J.len; ++c) {
                    dense[(I.start + r) * n + J.start + c] = bit(bits, r * J.len + c) ? 1 : 0;
                }
            }
            return;
        }
        const Eigen::Index i_mid = I.len / 2;
        const Eigen::Index j_mid = J.len / 2;
        const IndexRange I1{I.start, i_mid}, I2{I.start + i_mid, I.len - i_mid};
        const IndexRange J1{J.start, j_mid}, J2{J.start + j_mid, J.len - j_mid};
        expand(child(h, 0), I1, J1, dense);
        expand(child(h, 1), I1, J2, dense);
        expand(child(h, 2), I2, J1, dense);
        expand(child(h, 3), I2, J2, dense);
    }
};

namespace detail {

enum class MaskClass { Empty, Full, Partial };

/// Builds a quadtree from a pattern exposing classify(I, J) (a fast conservative test: it may
/// answer Partial when unsure) and allowed(i, j). Partial answers are resolved at the leaves and
/// collapsed upward, so classify only affects build time, never the result.
template <typename Pattern>
class QuadtreeBuilder {
public:
    QuadtreeBuilder(const Pattern& pattern, Eigen::Index n, Eigen::Index b) : pattern_(pattern), n_(n), b_(b) {}

    MaskQuadtree build() {
        MaskQuadtree tree;
        tree.n = n_;
        tree.b = b_;
        Fragment fragment;
        tree.root = build_block({0, n_}, {0, n_}, fragment, kParallelDepth);
        tree.nodes = std::move(fragment.nodes);
        tree.words = std::move(fragment.words);
        return tree;
    }

private:
    struct Fragment {
        std::vector<MaskQuadtree::Node> nodes;
        std::vector<uint64_t> words;
    };

    static constexpr int kParallelDepth = 4; // Top levels build their 4 children in parallel (256 tasks)

    const Pattern& pattern_;
    Eigen::Index n_;
    Eigen::Index b_;

    static int64_t handle_count(const Fragment& f, MaskHandle h, IndexRange I, IndexRange J) {
        if (h == kMaskFull) {
            return static_cast<int64_t>(I.len) * static_cast<int64_t>(J.len);
        }
        if (h == kMaskEmpty) {
            return 0;
        }
        return f.nodes[h].count;
    }

    /// Appends src to dst, shifting node and word references; returns src's handle in dst
    static MaskHandle splice(Fragment& dst, Fragment& src, MaskHandle h) {
        if (h < 0) {
            return h;
        }
        const auto node_offset = static_cast<MaskHandle>(dst.nodes.size());
        const auto word_offset = static_cast<int64_t>(dst.words.size());
        for (auto& node : src.nodes) {
            for (auto& c : node.child) {
                if (c >= 0) {
                    c += node_offset;
                }
            }
            if (node.bits >= 0) {
                node.bits += word_offset;
            }
        }
        dst.nodes.insert(dst.nodes.end(), src.nodes.begin(), src.nodes.end());
        dst.words.insert(dst.words.end(), src.words.begin(), src.words.end());
        return h + node_offset;
    }

    MaskHandle build_block(IndexRange I, IndexRange J, Fragment& f, int parallel_depth) {
        if (I.len <= 0 || J.len <= 0) {
            return kMaskEmpty;
        }
        const MaskClass cls = pattern_.classify(I, J);
        if (cls == MaskClass::Empty) {
            return kMaskEmpty;
        }
        if (cls == MaskClass::Full) {
            return kMaskFull;
        }
        const int64_t area = static_cast<int64_t>(I.len) * static_cast<int64_t>(J.len);

        if (I.len <= b_ && J.len <= b_) {
            // Leaf: evaluate every entry once
            std::vector<uint64_t> bits((area + 63) / 64, 0);
            int64_t count = 0;
            for (Eigen::Index r = 0; r < I.len; ++r) {
                for (Eigen::Index c = 0; c < J.len; ++c) {
                    if (pattern_.allowed(I.start + r, J.start + c)) {
                        const int64_t index = r * J.len + c;
                        bits[index >> 6] |= uint64_t{1} << (index & 63);
                        ++count;
                    }
                }
            }
            if (count == 0) {
                return kMaskEmpty;
            }
            if (count == area) {
                return kMaskFull;
            }
            MaskQuadtree::Node node;
            node.count = count;
            node.bits = static_cast<int64_t>(f.words.size());
            f.words.insert(f.words.end(), bits.begin(), bits.end());
            f.nodes.push_back(node);
            return static_cast<MaskHandle>(f.nodes.size() - 1);
        }

        const Eigen::Index i_mid = I.len / 2;
        const Eigen::Index j_mid = J.len / 2;
        const IndexRange quads_I[4] = {{I.start, i_mid}, {I.start, i_mid}, {I.start + i_mid, I.len - i_mid},
                                       {I.start + i_mid, I.len - i_mid}};
        const IndexRange quads_J[4] = {{J.start, j_mid}, {J.start + j_mid, J.len - j_mid}, {J.start, j_mid},
                                       {J.start + j_mid, J.len - j_mid}};

        MaskHandle h[4];
        if (parallel_depth > 0) {
            Fragment children[4];
            cilk_scope {
                for (int q = 0; q < 3; ++q) {
                    cilk_spawn [&, q] { h[q] = build_block(quads_I[q], quads_J[q], children[q], parallel_depth - 1); }();
                }
                h[3] = build_block(quads_I[3], quads_J[3], children[3], parallel_depth - 1);
            }
            for (int q = 0; q < 4; ++q) {
                h[q] = splice(f, children[q], h[q]);
            }
        } else {
            for (int q = 0; q < 4; ++q) {
                h[q] = build_block(quads_I[q], quads_J[q], f, 0);
            }
        }

        int64_t count = 0;
        for (int q = 0; q < 4; ++q) {
            count += handle_count(f, h[q], quads_I[q], quads_J[q]);
        }
        if (count == 0) {
            return kMaskEmpty;
        }
        if (count == area) {
            return kMaskFull;
        }
        MaskQuadtree::Node node;
        node.count = count;
        for (int q = 0; q < 4; ++q) {
            node.child[q] = h[q];
        }
        f.nodes.push_back(node);
        return static_cast<MaskHandle>(f.nodes.size() - 1);
    }
};

inline bool intervals_overlap(IndexRange I, IndexRange J) noexcept {
    return I.start < J.start + J.len && J.start < I.start + I.len;
}

/// Uniform [0, 1) value for a pair, independent of evaluation order
inline double pair_uniform(uint64_t seed, uint64_t a, uint64_t b) noexcept {
    uint64_t state = seed ^ (a * 0x9E3779B97F4A7C15ull) ^ (b * 0xC2B2AE3D27D4EB4Full);
    return static_cast<double>(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

/// Allowed pairs of a T x T tile grid: each tile kept with probability density (plus the diagonal)
struct BlockRandomPattern {
    Eigen::Index tile;
    double density;
    uint64_t seed;

    [[nodiscard]] bool tile_allowed(Eigen::Index ti, Eigen::Index tj) const noexcept {
        return pair_uniform(seed, static_cast<uint64_t>(ti), static_cast<uint64_t>(tj)) < density;
    }
    [[nodiscard]] bool allowed(Eigen::Index i, Eigen::Index j) const noexcept {
        return i == j || tile_allowed(i / tile, j / tile);
    }
    [[nodiscard]] MaskClass classify(IndexRange I, IndexRange J) const noexcept {
        const Eigen::Index ti0 = I.start / tile, ti1 = (I.start + I.len - 1) / tile;
        const Eigen::Index tj0 = J.start / tile, tj1 = (J.start + J.len - 1) / tile;
        if (ti0 == ti1 && tj0 == tj1) {
            if (tile_allowed(ti0, tj0)) {
                return MaskClass::Full;
            }
            return intervals_overlap(I, J) ? MaskClass::Partial : MaskClass::Empty;
        }
        return MaskClass::Partial;
    }
};

/// Sliding-window (banded) mask |i - j| <= width
struct BandPattern {
    Eigen::Index width;

    [[nodiscard]] bool allowed(Eigen::Index i, Eigen::Index j) const noexcept {
        const Eigen::Index diff = i - j;
        return diff <= width && -diff <= width;
    }
    [[nodiscard]] MaskClass classify(IndexRange I, IndexRange J) const noexcept {
        const Eigen::Index lo = I.start - (J.start + J.len - 1); // Smallest i - j in the block
        const Eigen::Index hi = (I.start + I.len - 1) - J.start; // Largest i - j in the block
        if (lo > width || hi < -width) {
            return MaskClass::Empty;
        }
        if (lo >= -width && hi <= width) {
            return MaskClass::Full;
        }
        return MaskClass::Partial;
    }
};

/// Independent random entries with probability density (plus the diagonal): the worst case,
/// where nearly every leaf is partial. Every entry is evaluated, so keep n modest (<= 2^17).
struct ElementRandomPattern {
    double density;
    uint64_t seed;

    [[nodiscard]] bool allowed(Eigen::Index i, Eigen::Index j) const noexcept {
        return i == j || pair_uniform(seed, static_cast<uint64_t>(i), static_cast<uint64_t>(j)) < density;
    }
    [[nodiscard]] MaskClass classify(IndexRange I, IndexRange J) const noexcept {
        if (density >= 1.0) {
            return MaskClass::Full;
        }
        if (density <= 0.0 && !intervals_overlap(I, J)) {
            return MaskClass::Empty;
        }
        return MaskClass::Partial;
    }
};

/// An explicit dense n x n 0/1 mask (for tests at small n); taken as given, without the diagonal
struct DensePattern {
    const std::vector<uint8_t>& mask;
    Eigen::Index n;

    [[nodiscard]] bool allowed(Eigen::Index i, Eigen::Index j) const noexcept { return mask[i * n + j] != 0; }
    [[nodiscard]] MaskClass classify(IndexRange, IndexRange) const noexcept { return MaskClass::Partial; }
};

} // namespace detail

template <typename Pattern>
inline MaskQuadtree build_mask_quadtree(const Pattern& pattern, Eigen::Index n, Eigen::Index b) {
    return detail::QuadtreeBuilder<Pattern>(pattern, n, b).build();
}

/// Builds a mask from a named pattern. density is the allowed-pair fraction (the diagonal is
/// always added so every row is nonempty, as Section 3.3.2.4 requires).
inline MaskQuadtree generate_mask(const std::string& pattern, Eigen::Index n, Eigen::Index b, double density,
                                  Eigen::Index tile, uint64_t seed) {
    if (pattern == "none") {
        return MaskQuadtree::full(n, b);
    }
    if (pattern == "block") {
        return build_mask_quadtree(detail::BlockRandomPattern{tile, density, seed}, n, b);
    }
    if (pattern == "band") {
        // Band half-width giving about density * n^2 allowed pairs (exact away from the corners)
        const auto width = static_cast<Eigen::Index>(std::max(0.0, (density * static_cast<double>(n) - 1.0) / 2.0));
        return build_mask_quadtree(detail::BandPattern{width}, n, b);
    }
    if (pattern == "element") {
        return build_mask_quadtree(detail::ElementRandomPattern{density, seed}, n, b);
    }
    throw std::runtime_error("Unknown mask pattern '" + pattern + "' (expected none, block, band or element)");
}
