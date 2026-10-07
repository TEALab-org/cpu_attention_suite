#include "attention_utilities.hpp" 
#include <cilk/cilk.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "standard_attention_parallel.hpp"
#include "flash_attention_2.hpp"
#include "rec_attention.hpp"

int main(int argc, char** argv) {
    std::string problem_path = (argc > 1) ? argv[1] : "data/problem_65536_64.npy";
    AttentionProblem problem(problem_path);
    const Eigen::Index N = problem.seq_len();

    std::cout << "Benchmarking attention for sequence length N = " << N << " (d = 64)\n";

    // 1. Run Standard Attention Reference
    auto t0 = std::chrono::high_resolution_clock::now();
    AttentionMatrixf O_ref = standard_attention_parallel(problem.Q, problem.K, problem.V, /*causal=*/true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_std = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Run FlashAttention-2
    auto t2 = std::chrono::high_resolution_clock::now();
    AttentionMatrixf O_fa2 = flash_attention_2(problem.Q, problem.K, problem.V, /*causal=*/true, 64, 64);
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_fa2 = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // 3. Verify Numerical Precision (L_inf and Frobenious norm delta)
    float max_diff = (O_ref - O_fa2).cwiseAbs().maxCoeff();
    float frob_norm = (O_ref - O_fa2).norm() / O_ref.norm();

    std::cout << "Standard Attention Time: " << t_std << " ms\n";
    std::cout << "FlashAttention-2 Time:   " << t_fa2 << " ms\n";
    std::cout << "Speedup:                 " << (t_std / t_fa2) << "x\n";
    std::cout << "Max Absolute Error:      " << max_diff << "\n";
    std::cout << "Relative Error:          " << frob_norm << "\n";

    return (max_diff < 1e-4f) ? 0 : 1;
}
