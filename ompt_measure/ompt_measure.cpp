#include <omp-tools.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct RegionInfo {
    int id = 0;
    unsigned long long executions = 0;
};

struct ParallelEvent {
    int region_id = 0;
    unsigned long long region_exec = 0;
    unsigned long long parallel_id = 0;
    const void *codeptr = nullptr;
    int requested_threads = 0;
    std::atomic<int> team_size{0};
    std::vector<double> start_ms;
    std::vector<double> current_start_ms;
    std::vector<double> end_ms;
    std::vector<double> elapsed_ms;
    std::vector<unsigned char> has_started;
    std::vector<unsigned char> work_started;
};

std::mutex state_mutex;
std::mutex output_mutex;
std::unordered_map<const void *, RegionInfo> region_by_codeptr;
std::atomic<unsigned long long> next_parallel_id{1};
Clock::time_point tool_start;
std::ofstream output;
int output_exec_interval = 1;

double now_ms() {
    const auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - tool_start).count();
}

bool has_thread_slot(const ParallelEvent *event, unsigned int thread_num) {
    return event && static_cast<std::size_t>(thread_num) < event->start_ms.size();
}

bool is_loop_work(ompt_work_t work_type) {
    return work_type == ompt_work_loop ||
           work_type == ompt_work_loop_static ||
           work_type == ompt_work_loop_dynamic ||
           work_type == ompt_work_loop_guided ||
           work_type == ompt_work_loop_other;
}

void record_thread_end(ParallelEvent *event, unsigned int thread_num) {
    if (!has_thread_slot(event, thread_num)) {
        return;
    }

    const std::size_t slot = static_cast<std::size_t>(thread_num);
    if (!event->work_started[slot]) {
        return;
    }

    event->end_ms[slot] = now_ms();
    event->elapsed_ms[slot] += event->end_ms[slot] - event->current_start_ms[slot];
    event->work_started[slot] = 0;
}

void write_header() {
    output << "parallel_id,region_id,region_exec,thread_id,team_size,"
           << "requested_threads,start_ms,end_ms,elapsed_ms,codeptr\n";
}

bool should_write_execution(const ParallelEvent &event) {
    return output_exec_interval <= 1 ||
           (event.region_exec > 0 && event.region_exec % output_exec_interval == 0);
}

}  // namespace

void on_parallel_begin(ompt_data_t *encountering_task_data,
                       const ompt_frame_t *encountering_task_frame,
                       ompt_data_t *parallel_data,
                       unsigned int requested_parallelism,
                       int flags,
                       const void *codeptr_ra) {
    (void)encountering_task_data;
    (void)encountering_task_frame;
    (void)flags;

    if (!parallel_data) {
        return;
    }

    auto *event = new ParallelEvent();
    event->parallel_id = next_parallel_id.fetch_add(1);
    event->codeptr = codeptr_ra;
    event->requested_threads = static_cast<int>(requested_parallelism);

    {
        std::lock_guard<std::mutex> lock(state_mutex);
        auto iter = region_by_codeptr.find(codeptr_ra);
        if (iter == region_by_codeptr.end()) {
            RegionInfo info;
            info.id = static_cast<int>(region_by_codeptr.size()) + 1;
            iter = region_by_codeptr.emplace(codeptr_ra, info).first;
        }

        RegionInfo &info = iter->second;
        info.executions += 1;
        event->region_id = info.id;
        event->region_exec = info.executions;
    }

    const unsigned int slots = std::max(1U,
                                        std::max(requested_parallelism,
                                                 static_cast<unsigned int>(omp_get_max_threads())));
    event->start_ms.assign(slots, 0.0);
    event->current_start_ms.assign(slots, 0.0);
    event->end_ms.assign(slots, 0.0);
    event->elapsed_ms.assign(slots, 0.0);
    event->has_started.assign(slots, 0);
    event->work_started.assign(slots, 0);

    parallel_data->ptr = event;
}

void on_work(ompt_work_t work_type,
             ompt_scope_endpoint_t endpoint,
             ompt_data_t *parallel_data,
             ompt_data_t *task_data,
             uint64_t count,
             const void *codeptr_ra) {
    (void)task_data;
    (void)count;
    (void)codeptr_ra;

    if (!is_loop_work(work_type) || !parallel_data || !parallel_data->ptr) {
        return;
    }

    auto *event = static_cast<ParallelEvent *>(parallel_data->ptr);
    const unsigned int thread_num = static_cast<unsigned int>(omp_get_thread_num());
    event->team_size.store(omp_get_num_threads(), std::memory_order_relaxed);
    if (!has_thread_slot(event, thread_num)) {
        return;
    }

    const std::size_t slot = static_cast<std::size_t>(thread_num);
    if (endpoint == ompt_scope_begin) {
        const double start = now_ms();
        if (!event->has_started[slot]) {
            event->start_ms[slot] = start;
            event->has_started[slot] = 1;
        }
        event->current_start_ms[slot] = start;
        event->work_started[slot] = 1;
    } else if (endpoint == ompt_scope_end) {
        record_thread_end(event, thread_num);
    }
}

void on_parallel_end(ompt_data_t *parallel_data,
                     ompt_data_t *task_data,
                     int flags,
                     const void *codeptr_ra) {
    (void)task_data;
    (void)flags;
    (void)codeptr_ra;

    if (!parallel_data || !parallel_data->ptr) {
        return;
    }

    auto *event = static_cast<ParallelEvent *>(parallel_data->ptr);
    if (!should_write_execution(*event)) {
        delete event;
        parallel_data->ptr = nullptr;
        return;
    }

    const int recorded_team_size = event->team_size.load(std::memory_order_relaxed);
    const int team_size = recorded_team_size > 0
                              ? std::min(recorded_team_size, static_cast<int>(event->elapsed_ms.size()))
                              : static_cast<int>(event->elapsed_ms.size());

    {
        std::lock_guard<std::mutex> lock(output_mutex);
        if (output) {
            for (int tid = 0; tid < team_size; ++tid) {
                const std::size_t index = static_cast<std::size_t>(tid);
                const double start = index < event->start_ms.size() ? event->start_ms[index] : 0.0;
                const double end = index < event->end_ms.size() ? event->end_ms[index] : 0.0;
                const double elapsed = index < event->elapsed_ms.size() ? event->elapsed_ms[index] : 0.0;

                output << event->parallel_id << ','
                       << event->region_id << ','
                       << event->region_exec << ','
                       << tid << ','
                       << team_size << ','
                       << event->requested_threads << ','
                       << std::fixed << std::setprecision(6)
                       << start << ','
                       << end << ','
                       << elapsed << ','
                       << event->codeptr << '\n';
            }
            output.flush();
        }
    }

    delete event;
    parallel_data->ptr = nullptr;
}

int ompt_initialize(ompt_function_lookup_t lookup,
                    int initial_device_num,
                    ompt_data_t *tool_data) {
    (void)initial_device_num;
    if (tool_data) {
        tool_data->value = 1ULL;
    }

    tool_start = Clock::now();

    const char *interval_text = std::getenv("AMR_OMPT_EXEC_INTERVAL");
    if (interval_text) {
        output_exec_interval = std::max(1, std::atoi(interval_text));
    }

    const char *output_path = std::getenv("AMR_OMPT_OUT");
    output.open(output_path ? output_path : "ompt_measure/ompt_regions.csv");
    if (!output) {
        std::fprintf(stderr, "[AMR-OMPT] failed to open output CSV\n");
        return 0;
    }
    write_header();

    auto set_callback = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (!set_callback) {
        std::fprintf(stderr, "[AMR-OMPT] ompt_set_callback is unavailable\n");
        return 0;
    }

    set_callback(ompt_callback_parallel_begin,
                 reinterpret_cast<ompt_callback_t>(&on_parallel_begin));
    set_callback(ompt_callback_parallel_end,
                 reinterpret_cast<ompt_callback_t>(&on_parallel_end));
    set_callback(ompt_callback_work,
                 reinterpret_cast<ompt_callback_t>(&on_work));

    std::fprintf(stderr, "[AMR-OMPT] enabled, writing %s every %d execution(s) per region\n",
                 output_path ? output_path : "ompt_measure/ompt_regions.csv",
                 output_exec_interval);
    return 1;
}

void ompt_finalize(ompt_data_t *tool_data) {
    (void)tool_data;
    std::lock_guard<std::mutex> lock(output_mutex);
    if (output) {
        output.flush();
        output.close();
    }
}

extern "C" {

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
    (void)omp_version;
    (void)runtime_version;
    static ompt_start_tool_result_t result = {&ompt_initialize, &ompt_finalize, {0ULL}};
    return &result;
}

}
