#include "attention_utilities.hpp" 
#include <iostream>
#include <iomanip>
#include <chrono>
#include <charconv>
#include <system_error>
#include "standard_attention_parallel.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
      std::cerr << "Error: expected 3 args, recieved " << argc << "\n";
      return 1;
    }
    std::string problem_path = argv[1];

    AttentionProblem problem(problem_path);
    const Eigen::Index N = problem.seq_len();

    std::cerr << "Benchmarking standard attention for sequence length N = " << N << " (d = 64)\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    AttentionMatrixf O_ref = standard_attention_parallel(problem.Q, problem.K, problem.V, /*causal=*/false);
;
    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout <<  std::setprecision(10) << duration << " ms\n";
    return 0;
}
