#include "ompt_test.hpp"
#include "../msr_freq.hpp"

#include <cstdio>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <omp.h>

namespace {

static bool debug = true;
static double max_mhz = 4800; // xeon w7-3565x

struct RegionState {
    const void* codeptr_ra;
    unsigned int begin_count;
    int nthreads;
    std::vector<std::chrono::steady_clock::time_point> start_times;
    std::vector<double> elapsed_ms;
    std::vector<double> target_mhz;
    std::vector<int> cpu_map;

    RegionState(const omptool::ParallelRegionBeginEvent& ev)
        : codeptr_ra(ev.codeptr_ra), begin_count(0) {
        nthreads = (ev.requested_parallelism > 0) ? static_cast<std::size_t>(ev.requested_parallelism) : static_cast<std::size_t>(omp_get_max_threads());
        elapsed_ms.assign(nthreads, 0.0);
        target_mhz.assign(nthreads, max_mhz);
        cpu_map.assign(nthreads, -1);
        start_times.assign(nthreads, std::chrono::steady_clock::time_point{});
    }

    void plan_balanced_freq() {
        if(debug) {
            std::fprintf(stderr, "[OMPT] balancing frequencies for region [%p] (begin_count=%u thread_count=%d)\n",
                        codeptr_ra, begin_count, nthreads);
        }

        if(begin_count == 1) {
            for(int i=0; i<nthreads; i++) target_mhz[i] = max_mhz;
            if(debug) {
                std::fprintf(stderr, "[OMPT]\t\t(First exec of region ==> %.0f MHz (MAX) for all threads)\n", max_mhz);
            }
            return;
        }

        double max_work = 0.0;
        for (int i = 0; i < nthreads; i++) {
            max_work = std::max(max_work, elapsed_ms[i]*target_mhz[i]);
        }
        
        for (int i = 0; i < nthreads; i++) {
            double work = elapsed_ms[i] * target_mhz[i];
            target_mhz[i] = max_mhz * (work / max_work);
            if(debug) {
                std::fprintf(stderr, "[OMPT]\t\t(Thread %d: previous elapsed: %.6f ms ==> %.0f MHz )\n",
                            i, elapsed_ms[i], target_mhz[i]);
            }
        }
    }
};

static std::unordered_map<const void*, RegionState> region_states;

}  // namespace



void dispatch_parallel_begin(ompt_data_t* encountering_task_data,
                             const ompt_frame_t* encountering_task_frame,
                             ompt_data_t* parallel_data,
                             unsigned int requested_parallelism,
                             int flags,
                             const void* codeptr_ra) {
    (void)encountering_task_frame;

    omptool::ParallelRegionBeginEvent event{
        encountering_task_data,
        parallel_data,
        requested_parallelism,
        flags,
        codeptr_ra,
    };

    if (debug) {
        std::fprintf(stderr, "[OMPT] begin region  [%p] (begin_count=%u thread_count=%d)\n",
                    event.codeptr_ra, rs->begin_count, rs->nthreads);
    }

    auto iter = region_states.find(event.codeptr_ra);
    if (iter == region_states.end()) {
        iter = region_states.emplace(event.codeptr_ra, RegionState(event)).first;
    }
    auto* rs = &iter->second;

    rs->begin_count += 1;
    rs->plan_balanced_freq();
    parallel_data->ptr = rs;
}


void dispatch_barrier_wait(ompt_sync_region_t kind,
                          ompt_scope_endpoint_t endpoint,
                          ompt_data_t* parallel_data,
                          ompt_data_t* task_data,
                          const void* codeptr_ra) {
    (void)kind; (void)task_data; (void)codeptr_ra;
    if (endpoint != ompt_scope_begin) return;
    if (!parallel_data || !parallel_data->ptr) {
        std::fprintf(stderr, "[OMPT] parallel_data is not registered\n");
        return;
    }

    auto* rs = static_cast<RegionState*>(parallel_data->ptr);
    if (!rs) {
        std::fprintf(stderr, "[OMPT] RegionState is not found\n");
        return;
    }

    int tid = omp_get_thread_num();
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(now - rs->start_times[tid]);
    rs->elapsed_ms[tid] = elapsed.count();

    if(rs->begin_count == 1) rs->cpu_map[tid] = msr::current_cpu();
}

void dispatch_implicit_task_begin(ompt_data_t* parallel_data,
                                  ompt_data_t* task_data,
                                  unsigned int team_size,
                                  unsigned int thread_num,
                                  int flags) {
    (void)task_data; (void)team_size; (void)flags;
    if (!parallel_data || !parallel_data->ptr) {
        std::fprintf(stderr, "[OMPT] implicit_task_begin: parallel_data is not registered\n");
        return;
    }

    auto* rs = static_cast<RegionState*>(parallel_data->ptr);
    if (!rs) {
        std::fprintf(stderr, "[OMPT] implicit_task_begin: RegionState is not found\n");
        return;
    }

    if(rs->begin_count == 1) rs->cpu_map[thread_num] = msr::current_cpu();
    
    msr::set_freq_on_cpu(rs->cpu_map[thread_num], rs->target_mhz[thread_num], 100);
    
    // 計測開始
    rs->start_times[thread_num] = std::chrono::steady_clock::now();
    task_data->ptr = rs;

    if(debug) {
        std::fprintf(stderr, "[OMPT] implicit_task_begin: thread %d on CPU %d, target %.0f MHz\n",
                    thread_num, rs->cpu_map[thread_num], rs->target_mhz[thread_num]);
    }
}

void dispatch_implicit_task_end(ompt_data_t* parallel_data,
                                ompt_data_t* task_data,
                                unsigned int team_size,
                                unsigned int thread_num,
                                int flags) {
    (void)task_data; (void)team_size; (void)flags;
    if (!parallel_data || !parallel_data->ptr) {
        std::fprintf(stderr, "[OMPT] implicit_task_end: parallel_data is not registered\n");
        return;
    }

    auto* rs = static_cast<RegionState*>(parallel_data->ptr);
    if (!rs) {
        std::fprintf(stderr, "[OMPT] implicit_task_end: RegionState is not found\n");
        return;
    }

    std::fprintf(stderr, "[OMPT] implicit_task_end: thread %d", thread_num);
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(now - rs->start_times[thread_num]);
    rs->elapsed_ms[thread_num] = elapsed.count();
    
    if(debug) {
        std::fprintf(stderr, "[OMPT] implicit_task_end: thread %d elapsed %.6f ms\n",
                    thread_num, rs->elapsed_ms[thread_num]);
    }
}

void dispatch_implicit_task(ompt_scope_endpoint_t endpoint,
                            ompt_data_t* parallel_data,
                            ompt_data_t* task_data,
                            unsigned int team_size,
                            unsigned int thread_num,
                            int flags) {
    if (endpoint == ompt_scope_begin) {
        dispatch_implicit_task_begin(parallel_data, task_data, team_size, thread_num, flags);
        return;
    }

    if (endpoint == ompt_scope_end) {
        dispatch_implicit_task_end(parallel_data, task_data, team_size, thread_num, flags);
    }
}

void dispatch_parallel_end(ompt_data_t* parallel_data,
                           ompt_data_t* task_data,
                           int flags,
                           const void* codeptr_ra) {
    (void)task_data; (void)flags; (void)codeptr_ra;
    if (debug) std::fprintf(stderr, "[OMPT] end region  [%p]\n", codeptr_ra);
    
    if (!parallel_data || !parallel_data->ptr) {
        std::fprintf(stderr, "[OMPT] parallel_data is not registered\n");
        return;
    }

    auto* rs = static_cast<RegionState*>(parallel_data->ptr);
    if (!rs) {
        std::fprintf(stderr, "[OMPT] RegionState is not found\n");
        return;
    }

    for(int i = 0; i < rs->nthreads; i++) {
        if(debug) std::fprintf(stderr, "[OMPT]\t\tResetting frequency on CPU %d\n", rs->cpu_map[i]);
        msr::set_freq_on_cpu(rs->cpu_map[i], 0, 100);
    }

    if(!debug) return;
    for (int i = 0; i < rs->nthreads; i++) {
        std::fprintf(stderr, "[OMPT]\t\t(thread %d: %.6f ms)\n",
                        i, rs->elapsed_ms[i]);
    }
}



int ompt_initialize(ompt_function_lookup_t lookup,
                    int /*initial_device_num*/,
                    ompt_data_t* tool_data) {
    if (tool_data) {
        tool_data->value = 1ULL;
    }

    auto set_callback = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (!set_callback) {
        std::fprintf(stderr, "[OMPT] ompt_set_callback is unavailable\n");
        return 0;
    }

    set_callback(ompt_callback_parallel_begin,
                 reinterpret_cast<ompt_callback_t>(&dispatch_parallel_begin));
    set_callback(ompt_callback_parallel_end,
                 reinterpret_cast<ompt_callback_t>(&dispatch_parallel_end));
    //set_callback(ompt_callback_sync_region,
    //             reinterpret_cast<ompt_callback_t>(&dispatch_barrier_wait));
    set_callback(ompt_callback_implicit_task,
                 reinterpret_cast<ompt_callback_t>(&dispatch_implicit_task));

    msr::msr_init();
    msr::reset_freq_all();
    return 1;
}

void ompt_finalize(ompt_data_t* /*tool_data*/) {
    (void)0;
}



extern "C" {

ompt_start_tool_result_t* ompt_start_tool(unsigned int /*omp_version*/,
                                         const char* /*runtime_version*/) {
    static ompt_start_tool_result_t result = {&ompt_initialize, &ompt_finalize, {0ULL}};
    return &result;
}

}
