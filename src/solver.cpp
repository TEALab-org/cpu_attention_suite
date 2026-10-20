#include "attention_utilities.hpp" 
#include <cilk/cilk.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "standard_attention_parallel.hpp"
#include "flash_attention_2.hpp"
//#include "dense_attention.hpp"
#include "parallel_summary.hpp"

int main(int argc, char** argv) {
    std::string problem_path = (argc > 1) ? argv[1] : "data/problem_65536_64.npy";
    AttentionProblem problem(problem_path);
    const Eigen::Index N = problem.seq_len();

    std::cout << "Benchmarking attention for sequence length N = " << N << " (d = 64)\n";

    // 1. Run Standard Attention Reference
    auto t0 = std::chrono::high_resolution_clock::now();
    AttentionMatrixf O_ref = standard_attention_parallel(problem.Q, problem.K, problem.V, /*causal=*/false);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_std = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << " Running Flash Attention" << std::endl;

    // 2. Run FlashAttention-2
    auto t2 = std::chrono::high_resolution_clock::now();
    AttentionMatrixf O_fa2 = flash_attention_2(problem.Q, problem.K, problem.V, /*causal=*/false, 64, 64);
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_fa2 = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << " Running Dense Attention" << std::endl;

    auto t4 = std::chrono::high_resolution_clock::now();
    //auto dro = dense_attention_recursive(problem.Q, problem.K, problem.V);
    AttentionMatrixf O_fa3 = cache_oblivious_attention_opencilk(problem.Q, problem.K, problem.V, 4);
    auto t5 = std::chrono::high_resolution_clock::now();
    double t_fa3 = std::chrono::duration<double, std::milli>(t5 - t4).count();


    // 3. Verify Numerical Precision (L_inf and Frobenious norm delta)
    float max_diff_1 = (O_ref - O_fa2).cwiseAbs().maxCoeff();
    float frob_norm_1 = (O_ref - O_fa2).norm() / O_ref.norm();

    float max_diff_2 = (O_ref - O_fa3).cwiseAbs().maxCoeff();
    float frob_norm_2 = (O_ref - O_fa3).norm() / O_ref.norm();


    std::cout << "Standard Attention Time: " << t_std << " ms\n";
    std::cout << "FlashAttention-2 Time:   " << t_fa2 << " ms\n";
    std::cout << "DenseAttention Time:   " << t_fa3 << " ms\n";
    std::cout << "Speedup 1:                 " << (t_std / t_fa2) << "x\n";
    std::cout << "Speedup 2:                 " << (t_std / t_fa3) << "x\n";
    std::cout << "FA2 Max Absolute Error:      " << max_diff_1 << "\n";
    std::cout << "FA2 Relative Error:          " << frob_norm_1 << "\n";
    std::cout << "DR Max Absolute Error:      " << max_diff_2 << "\n";
    std::cout << "DR Relative Error:          " << frob_norm_2 << "\n";


    return (max_diff_1 < 1e-4f) ? 0 : 1;
}
