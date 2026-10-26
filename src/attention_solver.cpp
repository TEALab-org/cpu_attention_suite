// Unified driver: dispatches to every dense attention solver, forward and backward.
//
//   attention_solver <algorithm> (<problem.npy> | --random N) [options]
//
// Backward algorithms (*_bwd) also need dO: a (4, N, 64) [Q, K, V, dO] problem file, or
// generated with --random. Their setup runs the FA2 forward once, untimed, for O and the
// row-wise log-sum-exp L; only the backward pass itself is timed and counted.
//
// Timings go to stdout: one "<t> ms" line per repetition (flushed as each finishes),
// one JSON object per repetition with --jsonl, or a single JSON summary with --json.
// Diagnostics go to stderr. Building with -DATTENTION_USE_PAPI adds per-repetition
// hardware counter totals (summed over all worker threads) to the JSON outputs.

#include "attention_utilities.hpp"
#include "dense_attention.hpp"
#include "flash_attention_2.hpp"
#include "flash_attention_2_backward.hpp"
#include "hybrid_attention.hpp"
#include "parallel_backward.hpp"
#include "parallel_summary.hpp"
#include "problem_generator.hpp"
#include "query_parallel.hpp"
#include "standard_attention_parallel.hpp"
#include "standard_attention_reference.hpp"
#include "standard_attention_streamed.hpp"
#include "standard_backward_parallel.hpp"

#include <cilk/cilk_api.h>
#ifdef ATTENTION_USE_PAPI
#include "papi_counters.hpp"
#endif
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
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
    std::string problem_path;
    Eigen::Index random_n = 0;    // > 0: generate a random problem instead of loading one
    uint64_t seed = 0;
    std::string output_path;
    bool json = false;
    bool jsonl = false;
    std::string papi_events = "PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_TCM";
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
        {"hybrid", "Summary recursion while p splits 4 ways, then query-parallel (uses -b, -g, -p)",
         [](const AttentionProblem& P, const SolverConfig& c) {
             return hybrid_attention(P.Q, P.K, P.V, c.p, c.block_size, c.grain);
         }},
    };
    return registry;
}

using BackwardFn = std::function<AttentionGradients(
    const AttentionProblem&, const AttentionMatrixf& O, const Eigen::VectorXf& L, const SolverConfig&)>;

struct BackwardEntry {
    const char* name;
    const char* description;
    BackwardFn run;
    bool needs_power_of_4_p = false;
};

const std::vector<BackwardEntry>& backward_registry() {
    static const std::vector<BackwardEntry> registry = {
        {"standard_bwd", "Parallel standard backward, materializes P and dS",
         [](const AttentionProblem& P, const AttentionMatrixf& O, const Eigen::VectorXf&, const SolverConfig&) {
             return standard_attention_backward_opencilk(P.Q, P.K, P.V, O, P.dO);
         }},
        {"fa2_bwd", "FlashAttention-2 backward, lock-free K/V and Q sweeps (uses -b)",
         [](const AttentionProblem& P, const AttentionMatrixf& O, const Eigen::VectorXf& L, const SolverConfig& c) {
             return flash_attention_2_backward_lockfree_opencilk(P.Q, P.K, P.V, O, P.dO, L, /*causal=*/false,
                                                                  c.block_size, c.block_size);
         }},
        {"summary_bwd", "ParallelDenseBackward, private gradient summaries (uses -b, -p)",
         [](const AttentionProblem& P, const AttentionMatrixf& O, const Eigen::VectorXf& L, const SolverConfig& c) {
             return parallel_dense_backward(P.Q, P.K, P.V, O, P.dO, L, c.p, c.block_size);
         },
         /*needs_power_of_4_p=*/true},
        {"owned_bwd", "ParallelOwnedBackward, two traversals with disjoint outputs (uses -b, -g)",
         [](const AttentionProblem& P, const AttentionMatrixf& O, const Eigen::VectorXf& L, const SolverConfig& c) {
             return parallel_owned_backward(P.Q, P.K, P.V, O, P.dO, L, c.block_size, c.grain);
         }},
        {"hybrid_bwd", "Summary recursion while p splits 4 ways, then owned (uses -b, -g, -p)",
         [](const AttentionProblem& P, const AttentionMatrixf& O, const Eigen::VectorXf& L, const SolverConfig& c) {
             return hybrid_backward(P.Q, P.K, P.V, O, P.dO, L, c.p, c.block_size, c.grain);
         }},
    };
    return registry;
}

const BackwardEntry* find_backward(std::string_view name) {
    for (const auto& entry : backward_registry()) {
        if (name == entry.name) {
            return &entry;
        }
    }
    return nullptr;
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
        << "Usage: " << argv0 << " <algorithm> (<problem.npy> | --random N) [options]\n\n"
        << "Forward algorithms:\n";
    for (const auto& entry : solver_registry()) {
        std::cerr << "  " << std::left << std::setw(13) << entry.name << entry.description << "\n";
    }
    std::cerr << "\nBackward algorithms:\n";
    for (const auto& entry : backward_registry()) {
        std::cerr << "  " << std::left << std::setw(13) << entry.name << entry.description << "\n";
    }
    std::cerr
        << "\nOptions:\n"
        << "      --random N      Generate a uniform [-1, 1) problem of length N instead of loading one\n"
        << "      --seed S        Seed for --random (default 0)\n"
        << "  -b, --block-size B  Tile / base case size (default 64)\n"
        << "  -g, --grain G       Query-parallel / owned serial cutoff (default 256)\n"
        << "  -p, --p P           Processor budget for summary* (power of 4) and hybrid* (default: nworkers)\n"
        << "  -r, --reps R        Timed repetitions (default 1)\n"
        << "  -w, --warmup W      Untimed warmup runs (default 0)\n"
        << "  -o, --output PATH   Write the last run's O, or stacked (3, N, 64) [dQ, dK, dV], to PATH (.npy)\n"
        << "      --json          Print a single JSON summary object after all repetitions\n"
        << "      --jsonl         Print one JSON object per repetition as it finishes\n"
#ifdef ATTENTION_USE_PAPI
        << "      --papi-events E Comma-separated PAPI events (default PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_TCM)\n"
#endif
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

[[maybe_unused]] std::vector<std::string> split_csv(const std::string& csv) {
    std::vector<std::string> items;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

struct RepResult {
    double time_ms = 0.0;
    std::vector<long long> counters; // Empty unless built with PAPI
};

void print_counters_json(std::ostream& out, const std::vector<std::string>& events, const RepResult& rep) {
    out << "{";
    for (size_t e = 0; e < events.size(); ++e) {
        out << (e ? ", " : "") << "\"" << events[e] << "\": " << rep.counters[e];
    }
    out << "}";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 2;
    }

    const SolverEntry* solver = find_solver(argv[1]);
    const BackwardEntry* backward = find_backward(argv[1]);
    const char* algorithm_name = solver ? solver->name : (backward ? backward->name : "");
    if (solver == nullptr && backward == nullptr) {
        std::cerr << "Error: unknown algorithm '" << argv[1] << "'\n\n";
        print_usage(argv[0]);
        return 2;
    }

    SolverConfig config;
    int first_option = 2;
    if (argv[2][0] != '-') {
        config.problem_path = argv[2];
        first_option = 3;
    }

    for (int i = first_option; i < argc; ++i) {
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
        } else if (arg == "--random" && has_value) {
            ok = parse_int(argv[++i], config.random_n) && config.random_n > 0;
        } else if (arg == "--seed" && has_value) {
            ok = parse_int(argv[++i], config.seed);
        } else if ((arg == "-o" || arg == "--output") && has_value) {
            config.output_path = argv[++i];
        } else if (arg == "--json") {
            config.json = true;
        } else if (arg == "--jsonl") {
            config.jsonl = true;
#ifdef ATTENTION_USE_PAPI
        } else if (arg == "--papi-events" && has_value) {
            config.papi_events = argv[++i];
            ok = !split_csv(config.papi_events).empty();
#endif
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

    if (config.problem_path.empty() == (config.random_n == 0)) {
        std::cerr << "Error: give exactly one of a problem file or --random N\n";
        return 2;
    }
    if (config.json && config.jsonl) {
        std::cerr << "Error: --json and --jsonl are mutually exclusive\n";
        return 2;
    }

    // Eigen lazily initializes a static cache-size table on first GEMM; touch it serially
    // so parallel solvers don't race on it (benign, but it pollutes Cilksan reports).
    (void)Eigen::l1CacheSize();

    const int nworkers = __cilkrts_get_nworkers();
    if (config.p == 0) {
        config.p = nworkers;
    }
    const bool needs_power_of_4_p = solver ? solver->needs_power_of_4_p : backward->needs_power_of_4_p;
    if (needs_power_of_4_p && !is_power_of_4(config.p)) {
        std::cerr << "Error: " << algorithm_name << " requires p to be a power of 4, got p = " << config.p
                  << (config.p == nworkers ? " (from nworkers; pass -p or set CILK_NWORKERS)" : "") << "\n";
        return 2;
    }

    AttentionProblem problem;
    try {
        problem = config.random_n > 0 ? generate_random_problem(config.random_n, config.seed, backward != nullptr)
                                      : AttentionProblem(config.problem_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    const Eigen::Index N = problem.seq_len();
    if (backward != nullptr && problem.dO.rows() != N) {
        std::cerr << "Error: " << backward->name << " needs dO: use a (4, N, 64) problem file or --random\n";
        return 2;
    }

    // Backward setup (untimed): one FA2 forward pass for O and the row-wise log-sum-exp L
    AttentionMatrixf O_forward;
    Eigen::VectorXf L_forward;
    if (backward != nullptr) {
        O_forward = flash_attention_2(problem.Q, problem.K, problem.V, /*causal=*/false, 64, 64, &L_forward);
    }

    std::vector<std::string> events;
#ifdef ATTENTION_USE_PAPI
    std::unique_ptr<ThreadCounters> counters;
    try {
        counters = std::make_unique<ThreadCounters>(split_csv(config.papi_events));
        events = counters->events();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
#endif

    std::cerr << "Running " << algorithm_name << " for N = " << N << " (d = 64), workers = " << nworkers
              << ", block size = " << config.block_size << ", grain = " << config.grain
              << ", p = " << config.p << "\n";

    AttentionMatrixf O;
    AttentionGradients grads;
    const auto solve = [&]() {
        if (backward != nullptr) {
            grads = backward->run(problem, O_forward, L_forward, config);
        } else {
            O = solver->run(problem, config);
        }
    };

    for (int w = 0; w < config.warmup; ++w) {
        solve();
    }

    std::cout << std::setprecision(10);
    std::vector<RepResult> reps;
    reps.reserve(config.reps);
    for (int r = 0; r < config.reps; ++r) {
        RepResult rep;
#ifdef ATTENTION_USE_PAPI
        counters->start();
#endif
        const auto t0 = std::chrono::steady_clock::now();
        solve();
        const auto t1 = std::chrono::steady_clock::now();
#ifdef ATTENTION_USE_PAPI
        rep.counters = counters->stop();
#endif
        rep.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Stream each repetition so a killed run keeps the repetitions it finished
        if (config.jsonl) {
            std::cout << "{\"rep\": " << r << ", \"time_ms\": " << rep.time_ms;
            if (!events.empty()) {
                std::cout << ", \"counters\": ";
                print_counters_json(std::cout, events, rep);
            }
            std::cout << "}" << std::endl;
        } else if (!config.json) {
            std::cout << rep.time_ms << " ms" << std::endl;
        }
        reps.push_back(std::move(rep));
    }

    if (!config.output_path.empty()) {
        try {
            if (backward != nullptr) {
                save_stacked_matrices_npy(config.output_path, {&grads.dQ, &grads.dK, &grads.dV});
            } else {
                save_attention_matrix_npy(config.output_path, O);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    }

    if (config.json) {
        std::cout << "{\"name\": \"" << algorithm_name << "\", \"N\": " << N << ", \"d\": 64"
                  << ", \"threads\": " << nworkers << ", \"block_size\": " << config.block_size
                  << ", \"base_size\": " << config.block_size << ", \"grain\": " << config.grain
                  << ", \"p\": " << config.p
                  << ", \"times\": [";
        for (size_t i = 0; i < reps.size(); ++i) {
            std::cout << (i ? ", " : "") << reps[i].time_ms;
        }
        std::cout << "]";
        if (!events.empty()) {
            std::cout << ", \"counters\": [";
            for (size_t i = 0; i < reps.size(); ++i) {
                std::cout << (i ? ", " : "");
                print_counters_json(std::cout, events, reps[i]);
            }
            std::cout << "]";
        }
        std::cout << "}\n";
    }
    return 0;
}
