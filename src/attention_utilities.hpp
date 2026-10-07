#pragma once

#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Eigen Type Aliases
// ============================================================================

template <typename Scalar = float>
using AttentionMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, 64, Eigen::RowMajor>;

using AttentionMatrixf = AttentionMatrix<float>;

template <typename Scalar = float>
using AttentionBlockMap = Eigen::Map<
    Eigen::Matrix<Scalar, Eigen::Dynamic, 64, Eigen::RowMajor>,
    Eigen::Aligned64
>;

// ============================================================================
// Internal .npy Parser Helpers
// ============================================================================

namespace detail {

inline std::string read_npy_header(std::ifstream& in) {
    char magic[6];
    in.read(magic, 6);
    if (!in || std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid file format: missing .npy magic string.");
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hlen16 = 0;
        in.read(reinterpret_cast<char*>(&hlen16), 2);
        header_len = hlen16;
    } else if (major == 2) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported .npy version: " + std::to_string(major));
    }

    std::string header(header_len, '\0');
    in.read(&header[0], header_len);
    if (!in) {
        throw std::runtime_error("Failed to read .npy header payload.");
    }

    return header;
}

inline std::vector<int64_t> parse_npy_shape(const std::string& header) {
    auto shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        shape_pos = header.find("\"shape\"");
    }
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("Malformed .npy header: 'shape' key not found.");
    }

    auto lparen = header.find('(', shape_pos);
    auto rparen = header.find(')', lparen);
    if (lparen == std::string::npos || rparen == std::string::npos) {
        throw std::runtime_error("Malformed .npy header: shape tuple delimiters not found.");
    }

    std::string tuple_str = header.substr(lparen + 1, rparen - lparen - 1);
    std::vector<int64_t> dims;
    std::stringstream ss(tuple_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        auto start = token.find_first_not_of(" \t\r\n");
        if (start != std::string::npos) {
            auto end = token.find_last_not_of(" \t\r\n");
            dims.push_back(std::stoll(token.substr(start, end - start + 1)));
        }
    }

    return dims;
}

inline void validate_npy_metadata(const std::string& header) {
    if (header.find("'fortran_order': True") != std::string::npos ||
        header.find("\"fortran_order\": True") != std::string::npos) {
        throw std::runtime_error("Fortran-ordered (column-major) .npy tensors are not supported.");
    }

    // Must be 32-bit floating point (float32)
    if (header.find("<f4") == std::string::npos &&
        header.find("=f4") == std::string::npos &&
        header.find("|f4") == std::string::npos) {
        throw std::runtime_error("Unsupported dtype: expected float32 ('<f4').");
    }
}

} // namespace detail

// ============================================================================
// Attention Problem Container
// ============================================================================

class AttentionProblem {
public:
    AttentionMatrixf Q;
    AttentionMatrixf K;
    AttentionMatrixf V;

    AttentionProblem() = default;

    /// Loads a 3D stacked (3, N, 64) float32 tensor from a .npy file.
    explicit AttentionProblem(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Unable to open attention problem file: " + filepath);
        }

        std::string header = detail::read_npy_header(in);
        detail::validate_npy_metadata(header);

        std::vector<int64_t> shape = detail::parse_npy_shape(header);
        if (shape.size() != 3 || shape[0] != 3 || shape[2] != 64) {
            std::string actual_shape = "(";
            for (size_t i = 0; i < shape.size(); ++i) {
                actual_shape += std::to_string(shape[i]) + (i + 1 < shape.size() ? ", " : ")");
            }
            throw std::runtime_error("Expected shape (3, N, 64), but encountered: " + actual_shape);
        }

        const int64_t N = shape[1];
        Q.resize(N, 64);
        K.resize(N, 64);
        V.resize(N, 64);

        const std::streamsize matrix_bytes = static_cast<std::streamsize>(N * 64 * sizeof(float));

        // In C-contiguous layout, Q, K, and V sit back-to-back in memory
        in.read(reinterpret_cast<char*>(Q.data()), matrix_bytes);
        in.read(reinterpret_cast<char*>(K.data()), matrix_bytes);
        in.read(reinterpret_cast<char*>(V.data()), matrix_bytes);

        if (!in) {
            throw std::runtime_error("Unexpected EOF while reading raw tensor payloads from: " + filepath);
        }
    }

    [[nodiscard]] Eigen::Index seq_len() const noexcept {
        return Q.rows();
    }
};

// ============================================================================
// Free Serialization Function
// ============================================================================

/// Saves a 2D AttentionMatrixf (N, 64) to a standard float32 row-major .npy file.
inline void save_attention_matrix_npy(const std::string& filepath, const AttentionMatrixf& mat) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to open destination file for writing: " + filepath);
    }

    const char magic[] = "\x93NUMPY";
    const uint8_t major = 1;
    const uint8_t minor = 0;

    std::string header_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
                            + std::to_string(mat.rows()) + ", "
                            + std::to_string(mat.cols()) + "), }";

    // Format alignment rule: (magic[6] + major[1] + minor[1] + header_len[2] + header) % 64 == 0
    constexpr size_t preamble_len = 10;
    const size_t min_header_len = header_dict.size() + 1; // +1 for trailing '\n'
    const size_t pad_len = (64 - ((preamble_len + min_header_len) % 64)) % 64;

    std::string header_padded = header_dict + std::string(pad_len, ' ') + "\n";
    const auto header_len = static_cast<uint16_t>(header_padded.size());

    out.write(magic, 6);
    out.write(reinterpret_cast<const char*>(&major), 1);
    out.write(reinterpret_cast<const char*>(&minor), 1);
    out.write(reinterpret_cast<const char*>(&header_len), 2);
    out.write(header_padded.data(), header_padded.size());

    // RowMajor matrices in Eigen are strictly C-contiguous
    out.write(reinterpret_cast<const char*>(mat.data()), mat.size() * sizeof(float));

    if (!out) {
        throw std::runtime_error("Error writing binary matrix buffer to: " + filepath);
    }
}
