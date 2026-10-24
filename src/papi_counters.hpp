#pragma once

// Hardware counter collection across all Cilk worker threads (build with -DATTENTION_USE_PAPI).
//
// A plain PAPI_start()/PAPI_stop() only counts the calling thread, and PAPI_INHERIT only
// covers threads created after counting starts. Cilk workers are long-lived threads that
// already exist, so instead we attach one EventSet to every thread of this process
// (enumerated from /proc/self/task) and start/stop them all from the main thread.

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <papi.h>

#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class ThreadCounters {
public:
    explicit ThreadCounters(std::vector<std::string> events) : events_(std::move(events)) {
        const int ver = PAPI_library_init(PAPI_VER_CURRENT);
        if (ver != PAPI_VER_CURRENT) {
            throw std::runtime_error("PAPI_library_init failed: " + papi_error(ver));
        }

        // Make sure the Cilk runtime and all its workers are up before enumerating threads
        const int nworkers = __cilkrts_get_nworkers();
        std::vector<long> busy(64 * nworkers);
        cilk_for (size_t i = 0; i < busy.size(); ++i) {
            long acc = 0;
            for (long k = 0; k < 10000; ++k) {
                acc += k ^ static_cast<long>(i);
            }
            busy[i] = acc;
        }
        if (std::find(busy.begin(), busy.end(), -1) != busy.end()) {
            std::cerr << ""; // Keeps the warm-up loop from being optimized away
        }

        const std::vector<pid_t> tids = list_threads();
        if (static_cast<int>(tids.size()) < nworkers) {
            std::cerr << "Warning: found " << tids.size() << " threads but " << nworkers
                      << " Cilk workers; counts may be incomplete\n";
        }

        for (pid_t tid : tids) {
            int es = PAPI_NULL;
            check(PAPI_create_eventset(&es), "PAPI_create_eventset");
            check(PAPI_assign_eventset_component(es, 0), "PAPI_assign_eventset_component");
            const int attach = PAPI_attach(es, static_cast<unsigned long>(tid));
            if (attach != PAPI_OK) {
                throw std::runtime_error("PAPI_attach to thread " + std::to_string(tid) + " failed: " +
                                         papi_error(attach) + " (kernel.perf_event_paranoid = " +
                                         read_paranoid() + "; per-thread user-space counting needs <= 2"
                                         " and hardware counters exposed to this node/VM)");
            }
            for (const auto& name : events_) {
                check(PAPI_add_named_event(es, name.c_str()), "PAPI_add_named_event(" + name + ")");
            }
            eventsets_.push_back(es);
        }
        std::cerr << "PAPI: counting " << events_.size() << " events on " << eventsets_.size()
                  << " threads\n";
    }

    ThreadCounters(const ThreadCounters&) = delete;
    ThreadCounters& operator=(const ThreadCounters&) = delete;

    ~ThreadCounters() {
        for (int es : eventsets_) {
            PAPI_cleanup_eventset(es);
            PAPI_destroy_eventset(&es);
        }
        PAPI_shutdown();
    }

    void start() {
        for (int es : eventsets_) {
            check(PAPI_start(es), "PAPI_start");
        }
    }

    /// Stops all EventSets and returns the per-event totals summed over threads.
    std::vector<long long> stop() {
        std::vector<long long> totals(events_.size(), 0);
        std::vector<long long> values(events_.size(), 0);
        for (int es : eventsets_) {
            check(PAPI_stop(es, values.data()), "PAPI_stop");
            for (size_t e = 0; e < values.size(); ++e) {
                totals[e] += values[e];
            }
        }
        return totals;
    }

    [[nodiscard]] const std::vector<std::string>& events() const noexcept { return events_; }

private:
    std::vector<std::string> events_;
    std::vector<int> eventsets_;

    static std::string papi_error(int code) {
        const char* msg = PAPI_strerror(code);
        return msg ? msg : ("error " + std::to_string(code));
    }

    static void check(int ret, const std::string& what) {
        if (ret != PAPI_OK) {
            throw std::runtime_error(what + " failed: " + papi_error(ret));
        }
    }

    static std::string read_paranoid() {
        std::ifstream in("/proc/sys/kernel/perf_event_paranoid");
        std::string value;
        return (in >> value) ? value : "unknown";
    }

    static std::vector<pid_t> list_threads() {
        std::vector<pid_t> tids;
        DIR* dir = opendir("/proc/self/task");
        if (dir == nullptr) {
            throw std::runtime_error("Unable to open /proc/self/task");
        }
        while (dirent* entry = readdir(dir)) {
            if (entry->d_name[0] != '.') {
                tids.push_back(static_cast<pid_t>(std::atol(entry->d_name)));
            }
        }
        closedir(dir);
        std::sort(tids.begin(), tids.end());
        return tids;
    }
};
