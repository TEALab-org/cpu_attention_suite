#include "attention_utilities.hpp" 
#include <cilk/cilk_api.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <charconv>
#include <system_error>
#include "parallel_summary.hpp"


int main(int argc, char** argv) {
    if (argc != 3) {
      std::cerr << "Error: expected 3 args, recieved " << argc << "\n";
      return 1;
    }
    std::string problem_path = argv[1];

    // Parse base_size
    Eigen::Index base_size = 0;
    const std::string_view str(argv[2]);
    auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), base_size);
    if (ec == std::errc::invalid_argument) {
        std::cerr << "Error: Not a valid integer.\n";
        return 1;
    } else if (ec == std::errc::result_out_of_range) {
        std::cerr << "Error: Value out of range for Eigen::Index.\n";
        return 1;
    } else if (ptr != str.data() + str.size()) {
        std::cerr << "Error: Trailing characters detected after index.\n";
        return 1;
    }

    auto const nworkers = __cilkrts_get_nworkers();

    AttentionProblem problem(problem_path);
    const Eigen::Index N = problem.seq_len();

    std::cerr << "Benchmarking rec attention 2 for sequence length N = " << N << " (d = 64), N workers: " << nworkers << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    AttentionMatrixf O_fa2 = cache_oblivious_attention_opencilk(problem.Q, problem.K, problem.V, nworkers, base_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout <<  std::setprecision(10) << duration << " ms\n";
    return 0;
}
