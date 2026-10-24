// Unified forward-pass driver: dispatches to every dense attention solver.
//
//   attention_solver <algorithm> <problem.npy> [options]
//
// Timings go to stdout (one "<t> ms" line per repetition, or a single JSON
// object with --json); diagnostics go to stderr.

#include "attention_utilities.hpp"
#include "dense_attention.hpp"
#include "flash_attention_2.hpp"
#include "parallel_summary.hpp"
#include "query_parallel.hpp"
#include "standard_attention_parallel.hpp"
#include "standard_attention_reference.hpp"
#include "standard_attention_streamed.hpp"

#include <cilk/cilk_api.h>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct SolverConfig {
    Eigen::Index block_size = 64; // FA2 tile size (Br = Bc) / recursive base case size
    Eigen::Index grain = 256;     // QueryParallel serial cutoff on |I|
    int p = 0;                    // Parallel summary processor budget (0 = nworkers)
    int reps = 1;
    int warmup = 0;
    std::string output_path;
    bool json = false;
};

using SolverFn = std::function<AttentionMatrixf(const AttentionProblem&, const SolverConfig&)>;

struct SolverEntry {
    const char* name;
    const char* description;
    SolverFn run;
    bool needs_power_of_4_p = false;
};

const std::vector<SolverEntry>& solver_registry() {
    static const std::vector<SolverEntry> registry = {
        {"reference", "Serial standard attention, materializes S (ground truth)",
         [](const AttentionProblem& P, const SolverConfig&) {
             return standard_attention_reference(P.Q, P.K, P.V);
         }},
        {"standard", "Parallel standard attention, materializes S",
         [](const AttentionProblem& P, const SolverConfig&) {
             return standard_attention_parallel(P.Q, P.K, P.V);
         }},
        {"streamed", "Parallel row-streamed standard attention",
         [](const AttentionProblem& P, const SolverConfig&) {
             return standard_attention_streamed_opencilk(P.Q, P.K, P.V);
         }},
        {"fa2", "FlashAttention-2, parallel over query blocks (uses -b)",
         [](const AttentionProblem& P, const SolverConfig& c) {
             return flash_attention_2(P.Q, P.K, P.V, /*causal=*/false, c.block_size, c.block_size);
         }},
        {"recursive", "Serial cache-oblivious recursive attention (uses -b)",
         [](const AttentionProblem& P, const SolverConfig& c) {
             return dense_attention_recursive(P.Q, P.K, P.V, c.block_size).O;
         }},
        {"summary", "Parallel summary cache-oblivious attention (uses -b, -p)",
         [](const AttentionProblem& P, const SolverConfig& c) {
             return cache_oblivious_attention_opencilk(P.Q, P.K, P.V, c.p, c.block_size);
         },
         /*needs_power_of_4_p=*/true},
        {"query", "Query-parallel cache-oblivious attention (uses -b, -g)",
         [](const AttentionProblem& P, const SolverConfig& c) {
             return query_parallel_attention(P.Q, P.K, P.V, c.block_size, c.grain);
         }},
    };
    return registry;
}

const SolverEntry* find_solver(std::string_view name) {
    if (name == "rec") {
        name = "summary"; // Legacy name from rec_solver
    }
    for (const auto& entry : solver_registry()) {
        if (name == entry.name) {
            return &entry;
        }
    }
    return nullptr;
}

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <algorithm> <problem.npy> [options]\n\n"
        << "Algorithms:\n";
    for (const auto& entry : solver_registry()) {
        std::cerr << "  " << std::left << std::setw(11) << entry.name << entry.description << "\n";
    }
    std::cerr
        << "\nOptions:\n"
        << "  -b, --block-size B  Tile / base case size (default 64)\n"
        << "  -g, --grain G       Query-parallel serial cutoff on |I| (default 256)\n"
        << "  -p, --p P           Summary processor budget, must be a power of 4 (default: nworkers)\n"
        << "  -r, --reps R        Timed repetitions (default 1)\n"
        << "  -w, --warmup W      Untimed warmup runs (default 0)\n"
        << "  -o, --output PATH   Write the output matrix O of the last run to PATH (.npy)\n"
        << "      --json          Print a single JSON result object instead of per-rep times\n"
        << "\nWorker count is controlled by the CILK_NWORKERS environment variable.\n";
}

template <typename Int>
bool parse_int(std::string_view str, Int& out) {
    auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), out);
    return ec == std::errc() && ptr == str.data() + str.size();
}

bool is_power_of_4(int p) {
    return p >= 1 && (p & (p - 1)) == 0 && (p & 0x55555555) != 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 2;
    }

    const SolverEntry* solver = find_solver(argv[1]);
    if (solver == nullptr) {
        std::cerr << "Error: unknown algorithm '" << argv[1] << "'\n\n";
        print_usage(argv[0]);
        return 2;
    }
    const std::string problem_path = argv[2];

    SolverConfig config;
    for (int i = 3; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const bool has_value = (i + 1 < argc);
        bool ok = true;

        if ((arg == "-b" || arg == "--block-size") && has_value) {
            ok = parse_int(argv[++i], config.block_size) && config.block_size > 0;
        } else if ((arg == "-p" || arg == "--p") && has_value) {
            ok = parse_int(argv[++i], config.p) && config.p > 0;
        } else if ((arg == "-g" || arg == "--grain") && has_value) {
            ok = parse_int(argv[++i], config.grain) && config.grain > 0;
        } else if ((arg == "-r" || arg == "--reps") && has_value) {
            ok = parse_int(argv[++i], config.reps) && config.reps > 0;
        } else if ((arg == "-w" || arg == "--warmup") && has_value) {
            ok = parse_int(argv[++i], config.warmup) && config.warmup >= 0;
        } else if ((arg == "-o" || arg == "--output") && has_value) {
            config.output_path = argv[++i];
        } else if (arg == "--json") {
            config.json = true;
        } else {
            std::cerr << "Error: unrecognized or incomplete option '" << arg << "'\n\n";
            print_usage(argv[0]);
            return 2;
        }

        if (!ok) {
            std::cerr << "Error: invalid value for " << arg << ": '" << argv[i] << "'\n";
            return 2;
        }
    }

    // Eigen lazily initializes a static cache-size table on first GEMM; touch it serially
    // so parallel solvers don't race on it (benign, but it pollutes Cilksan reports).
    (void)Eigen::l1CacheSize();

    const int nworkers = __cilkrts_get_nworkers();
    if (config.p == 0) {
        config.p = nworkers;
    }
    if (solver->needs_power_of_4_p && !is_power_of_4(config.p)) {
        std::cerr << "Error: " << solver->name << " requires p to be a power of 4, got p = " << config.p
                  << (config.p == nworkers ? " (from nworkers; pass -p or set CILK_NWORKERS)" : "") << "\n";
        return 2;
    }

    AttentionProblem problem;
    try {
        problem = AttentionProblem(problem_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    const Eigen::Index N = problem.seq_len();

    std::cerr << "Running " << solver->name << " for N = " << N << " (d = 64), workers = " << nworkers
              << ", block size = " << config.block_size << ", grain = " << config.grain
              << ", p = " << config.p << "\n";

    AttentionMatrixf O;
    for (int w = 0; w < config.warmup; ++w) {
        O = solver->run(problem, config);
    }

    std::vector<double> times;
    times.reserve(config.reps);
    for (int r = 0; r < config.reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        O = solver->run(problem, config);
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    if (!config.output_path.empty()) {
        try {
            save_attention_matrix_npy(config.output_path, O);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    }

    std::cout << std::setprecision(10);
    if (config.json) {
        std::cout << "{\"name\": \"" << solver->name << "\", \"N\": " << N << ", \"d\": 64"
                  << ", \"threads\": " << nworkers << ", \"block_size\": " << config.block_size
                  << ", \"base_size\": " << config.block_size << ", \"grain\": " << config.grain
                  << ", \"p\": " << config.p
                  << ", \"times\": [";
        for (size_t i = 0; i < times.size(); ++i) {
            std::cout << (i ? ", " : "") << times[i];
        }
        std::cout << "]}\n";
    } else {
        for (double t : times) {
            std::cout << t << " ms\n";
        }
    }
    return 0;
}
