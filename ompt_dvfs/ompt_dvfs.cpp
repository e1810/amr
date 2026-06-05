#include <omp-tools.h>
#include <omp.h>

#include "msr_freq.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

struct RegionState {
    int id = 0;
    unsigned long long executions = 0;
    bool dvfs_enabled = false;
    double last_region_wall_ms = 0.0;
    std::vector<double> last_elapsed_ms;
    std::vector<double> last_target_mhz;
};

struct ParallelEvent {
    int region_id = 0;
    unsigned long long region_exec = 0;
    unsigned long long parallel_id = 0;
    const void *codeptr = nullptr;
    int requested_threads = 0;
    bool dvfs_enabled = false;
    double region_wall_ms = 0.0;
    std::atomic<int> team_size{0};
    std::vector<double> start_ms;
    std::vector<double> current_start_ms;
    std::vector<double> end_ms;
    std::vector<double> elapsed_ms;
    std::vector<double> target_mhz;
    std::vector<msr::CounterSample> current_sample;
    std::vector<std::uint64_t> aperf_delta;
    std::vector<std::uint64_t> mperf_delta;
    std::vector<int> cpu_id;
    std::vector<int> sample_cpu_id;
    std::vector<unsigned char> has_started;
    std::vector<unsigned char> work_started;
    std::vector<unsigned char> sample_started;
    std::vector<unsigned char> resource_applied;
};

std::mutex state_mutex;
std::mutex output_mutex;
std::unordered_map<const void *, RegionState> region_by_codeptr;
std::atomic<unsigned long long> next_parallel_id{1};
Clock::time_point tool_start;
std::ofstream output;
int output_exec_interval = 1;
double max_mhz = 4800.0;
double min_mhz = 800.0;
double bus_mhz = 100.0;
double reference_mhz = 0.0;
double dvfs_enable_region_ms = 1.2;
double dvfs_disable_region_ms = 0.8;
bool debug = false;
bool msr_control_enabled = false;
bool msr_counter_enabled = false;

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

double env_double(const char *name, double fallback) {
    const char *text = std::getenv(name);
    if (!text) {
        return fallback;
    }

    char *end = nullptr;
    const double value = std::strtod(text, &end);
    return end != text ? value : fallback;
}

bool env_bool(const char *name, bool fallback) {
    const char *text = std::getenv(name);
    if (!text) {
        return fallback;
    }

    return std::atoi(text) != 0;
}

bool can_open_current_msr(int flags) {
    const int cpu = msr::current_cpu();
    if (cpu < 0) {
        return false;
    }

    char path[64];
    std::snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu);
    const int fd = open(path, flags);
    if (fd < 0) {
        return false;
    }

    close(fd);
    return true;
}

bool can_open_current_msr_for_read() {
    return can_open_current_msr(O_RDONLY);
}

bool can_open_current_msr_for_write() {
    return can_open_current_msr(O_WRONLY);
}

bool valid_counter_sample(const msr::CounterSample &sample) {
    return sample.aperf != 0ULL && sample.mperf != 0ULL;
}

double reference_mhz_for_cpu(int cpu) {
    if (reference_mhz > 0.0) {
        return reference_mhz;
    }
    return msr::platform_base_mhz_on_cpu(cpu, bus_mhz);
}

void begin_thread_sample(ParallelEvent *event, std::size_t slot, int cpu) {
    if (!event->dvfs_enabled || !msr_counter_enabled || cpu < 0 ||
        slot >= event->current_sample.size()) {
        return;
    }

    const msr::CounterSample sample = msr::sample_on_cpu(cpu);
    if (!valid_counter_sample(sample)) {
        return;
    }

    event->current_sample[slot] = sample;
    event->sample_cpu_id[slot] = cpu;
    event->sample_started[slot] = 1;
}

void end_thread_sample(ParallelEvent *event, std::size_t slot) {
    if (!event->dvfs_enabled || !msr_counter_enabled ||
        slot >= event->sample_started.size() ||
        !event->sample_started[slot]) {
        return;
    }

    event->sample_started[slot] = 0;
    const int cpu = msr::current_cpu();
    if (slot >= event->sample_cpu_id.size() || cpu != event->sample_cpu_id[slot]) {
        return;
    }

    const msr::CounterSample first = event->current_sample[slot];
    const msr::CounterSample second = msr::sample_on_cpu(cpu);
    if (second.aperf <= first.aperf || second.mperf <= first.mperf) {
        return;
    }

    event->aperf_delta[slot] += second.aperf - first.aperf;
    event->mperf_delta[slot] += second.mperf - first.mperf;
}

double measured_mhz_for_thread(const ParallelEvent &event, std::size_t slot) {
    if (slot >= event.aperf_delta.size() || slot >= event.mperf_delta.size() ||
        event.aperf_delta[slot] == 0ULL || event.mperf_delta[slot] == 0ULL) {
        return -1.0;
    }

    const int cpu = slot < event.sample_cpu_id.size() ? event.sample_cpu_id[slot] : -1;
    const double base_mhz = reference_mhz_for_cpu(cpu);
    if (base_mhz <= 0.0) {
        return -1.0;
    }

    const msr::CounterSample first{0ULL, 0ULL};
    const msr::CounterSample second{event.aperf_delta[slot], event.mperf_delta[slot]};
    return msr::compute_freq_mhz(base_mhz, first, second);
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
    end_thread_sample(event, slot);
    event->work_started[slot] = 0;
}

std::vector<double> plan_targets(const RegionState &region, std::size_t slots) {
    std::vector<double> targets(slots, max_mhz);
    //return targets; // set all max
    if (region.executions <= 1 || region.last_elapsed_ms.empty()) {
        return targets;
    }

    double max_work = 0.0;
    for (std::size_t i = 0; i < slots && i < region.last_elapsed_ms.size(); ++i) {
        const double previous_target =
            i < region.last_target_mhz.size() && region.last_target_mhz[i] > 0.0
                ? region.last_target_mhz[i]
                : max_mhz;
        max_work = std::max(max_work, region.last_elapsed_ms[i] * previous_target);
    }
    if (max_work <= 0.0) {
        return targets;
    }

    for (std::size_t i = 0; i < slots && i < region.last_elapsed_ms.size(); ++i) {
        const double previous_target =
            i < region.last_target_mhz.size() && region.last_target_mhz[i] > 0.0
                ? region.last_target_mhz[i]
                : max_mhz;
        const double work = region.last_elapsed_ms[i] * previous_target;
        targets[i] = std::clamp(max_mhz * (work / max_work), min_mhz, max_mhz);
    }
    return targets;
}

void write_header() {
    output << "parallel_id,region_id,region_exec,thread_id,team_size,"
           << "requested_threads,start_ms,end_ms,elapsed_ms,codeptr,"
           << "cpu_id,target_mhz,measured_mhz,resource_applied,"
           << "dvfs_enabled,region_wall_ms,dvfs_enable_ms,dvfs_disable_ms\n";
}

bool should_write_execution(const ParallelEvent &event) {
    return output_exec_interval <= 1 ||
           (event.region_exec > 0 && event.region_exec % output_exec_interval == 0);
}

int event_team_size(const ParallelEvent &event) {
    const int recorded_team_size = event.team_size.load(std::memory_order_relaxed);
    return recorded_team_size > 0
               ? std::min(recorded_team_size, static_cast<int>(event.elapsed_ms.size()))
               : static_cast<int>(event.elapsed_ms.size());
}

void apply_target_to_thread(ParallelEvent *event, unsigned int thread_num) {
    if (!has_thread_slot(event, thread_num)) {
        return;
    }

    const std::size_t slot = static_cast<std::size_t>(thread_num);
    const int cpu = msr::current_cpu();
    event->cpu_id[slot] = cpu;
    if (cpu < 0) {
        return;
    }

    if (!event->dvfs_enabled || !msr_control_enabled) {
        return;
    }

    const bool applied = msr::set_freq_on_cpu(cpu, event->target_mhz[slot], 100.0);
    event->resource_applied[slot] = applied ? 1 : 0;

    if (debug) {
        std::fprintf(stderr,
                     "[AMR-DVFS] region=%d exec=%llu thread=%u cpu=%d target=%.0fMHz %s\n",
                     event->region_id,
                     event->region_exec,
                     thread_num,
                     cpu,
                     event->target_mhz[slot],
                     applied ? "applied" : "failed");
    }
}

double compute_region_wall_ms(const ParallelEvent &event, int team_size) {
    bool found = false;
    double min_start = 0.0;
    double max_end = 0.0;
    for (int tid = 0; tid < team_size; ++tid) {
        const std::size_t index = static_cast<std::size_t>(tid);
        if (index >= event.has_started.size() || index >= event.start_ms.size() ||
            index >= event.end_ms.size() || !event.has_started[index]) {
            continue;
        }

        if (!found) {
            min_start = event.start_ms[index];
            max_end = event.end_ms[index];
            found = true;
            continue;
        }

        min_start = std::min(min_start, event.start_ms[index]);
        max_end = std::max(max_end, event.end_ms[index]);
    }

    return found && max_end >= min_start ? max_end - min_start : 0.0;
}

void reset_touched_resources(const ParallelEvent &event, int team_size) {
    std::vector<int> reset_cpus;
    for (int tid = 0; tid < team_size; ++tid) {
        const std::size_t index = static_cast<std::size_t>(tid);
        if (index >= event.cpu_id.size() || index >= event.resource_applied.size() ||
            !event.resource_applied[index] || event.cpu_id[index] < 0) {
            continue;
        }

        const int cpu = event.cpu_id[index];
        if (std::find(reset_cpus.begin(), reset_cpus.end(), cpu) == reset_cpus.end()) {
            reset_cpus.push_back(cpu);
            msr::set_freq_on_cpu(cpu, 0.0, 100.0);
        }
    }
}

void update_region_history(const ParallelEvent &event, int team_size) {
    std::lock_guard<std::mutex> lock(state_mutex);
    auto iter = region_by_codeptr.find(event.codeptr);
    if (iter == region_by_codeptr.end()) {
        return;
    }

    RegionState &region = iter->second;
    region.last_elapsed_ms.assign(static_cast<std::size_t>(team_size), 0.0);
    region.last_target_mhz.assign(static_cast<std::size_t>(team_size), max_mhz);
    for (int tid = 0; tid < team_size; ++tid) {
        const std::size_t index = static_cast<std::size_t>(tid);
        region.last_elapsed_ms[index] =
            index < event.elapsed_ms.size() ? event.elapsed_ms[index] : 0.0;
        region.last_target_mhz[index] =
            index < event.target_mhz.size() ? event.target_mhz[index] : max_mhz;
    }

    region.last_region_wall_ms = event.region_wall_ms;
    if (!region.dvfs_enabled && event.region_wall_ms >= dvfs_enable_region_ms) {
        region.dvfs_enabled = true;
    } else if (region.dvfs_enabled && event.region_wall_ms <= dvfs_disable_region_ms) {
        region.dvfs_enabled = false;
    }
}

void write_event_csv(const ParallelEvent &event, int team_size) {
    std::lock_guard<std::mutex> lock(output_mutex);
    if (!output) {
        return;
    }

    for (int tid = 0; tid < team_size; ++tid) {
        const std::size_t index = static_cast<std::size_t>(tid);
        const double start = index < event.start_ms.size() ? event.start_ms[index] : 0.0;
        const double end = index < event.end_ms.size() ? event.end_ms[index] : 0.0;
        const double elapsed = index < event.elapsed_ms.size() ? event.elapsed_ms[index] : 0.0;
        const int cpu = index < event.cpu_id.size() ? event.cpu_id[index] : -1;
        const double target = index < event.target_mhz.size() ? event.target_mhz[index] : max_mhz;
        const double measured = measured_mhz_for_thread(event, index);
        const int applied =
            index < event.resource_applied.size() ? static_cast<int>(event.resource_applied[index]) : 0;

        output << event.parallel_id << ','
               << event.region_id << ','
               << event.region_exec << ','
               << tid << ','
               << team_size << ','
               << event.requested_threads << ','
               << std::fixed << std::setprecision(6)
               << start << ','
               << end << ','
               << elapsed << ','
               << event.codeptr << ','
               << cpu << ','
               << target << ','
               << measured << ','
               << applied << ','
               << (event.dvfs_enabled ? 1 : 0) << ','
               << event.region_wall_ms << ','
               << dvfs_enable_region_ms << ','
               << dvfs_disable_region_ms << '\n';
    }
    output.flush();
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

    const unsigned int slots = std::max(1U,
                                        std::max(requested_parallelism,
                                                 static_cast<unsigned int>(omp_get_max_threads())));
    event->start_ms.assign(slots, 0.0);
    event->current_start_ms.assign(slots, 0.0);
    event->end_ms.assign(slots, 0.0);
    event->elapsed_ms.assign(slots, 0.0);
    event->current_sample.assign(slots, msr::CounterSample{0ULL, 0ULL});
    event->aperf_delta.assign(slots, 0ULL);
    event->mperf_delta.assign(slots, 0ULL);
    event->cpu_id.assign(slots, -1);
    event->sample_cpu_id.assign(slots, -1);
    event->has_started.assign(slots, 0);
    event->work_started.assign(slots, 0);
    event->sample_started.assign(slots, 0);
    event->resource_applied.assign(slots, 0);

    {
        std::lock_guard<std::mutex> lock(state_mutex);
        auto iter = region_by_codeptr.find(codeptr_ra);
        if (iter == region_by_codeptr.end()) {
            RegionState state;
            state.id = static_cast<int>(region_by_codeptr.size()) + 1;
            iter = region_by_codeptr.emplace(codeptr_ra, state).first;
        }

        RegionState &state = iter->second;
        state.executions += 1;
        event->region_id = state.id;
        event->region_exec = state.executions;
        event->dvfs_enabled = state.dvfs_enabled;
        event->target_mhz = event->dvfs_enabled
                                ? plan_targets(state, static_cast<std::size_t>(slots))
                                : std::vector<double>(static_cast<std::size_t>(slots), max_mhz);
    }

    parallel_data->ptr = event;
}

void on_implicit_task(ompt_scope_endpoint_t endpoint,
                      ompt_data_t *parallel_data,
                      ompt_data_t *task_data,
                      unsigned int team_size,
                      unsigned int thread_num,
                      int flags) {
    (void)task_data;
    (void)flags;

    if (endpoint != ompt_scope_begin || !parallel_data || !parallel_data->ptr) {
        return;
    }

    auto *event = static_cast<ParallelEvent *>(parallel_data->ptr);
    event->team_size.store(static_cast<int>(team_size), std::memory_order_relaxed);
    apply_target_to_thread(event, thread_num);
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
        begin_thread_sample(event, slot, msr::current_cpu());
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
    for (unsigned int tid = 0; tid < event->work_started.size(); ++tid) {
        record_thread_end(event, tid);
    }

    const int team_size = event_team_size(*event);
    event->region_wall_ms = compute_region_wall_ms(*event, team_size);
    reset_touched_resources(*event, team_size);
    update_region_history(*event, team_size);

    if (should_write_execution(*event)) {
        write_event_csv(*event, team_size);
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
    max_mhz = env_double("AMR_DVFS_MAX_MHZ", env_double("AMR_RESCTRL_MAX_MHZ", max_mhz));
    min_mhz = env_double("AMR_DVFS_MIN_MHZ", env_double("AMR_RESCTRL_MIN_MHZ", min_mhz));
    bus_mhz = env_double("AMR_DVFS_BUS_MHZ", bus_mhz);
    reference_mhz = env_double("AMR_DVFS_BASE_MHZ", reference_mhz);
    dvfs_enable_region_ms = env_double("AMR_DVFS_ENABLE_REGION_MS", dvfs_enable_region_ms);
    dvfs_disable_region_ms = env_double("AMR_DVFS_DISABLE_REGION_MS", dvfs_disable_region_ms);
    if (dvfs_disable_region_ms > dvfs_enable_region_ms) {
        std::swap(dvfs_disable_region_ms, dvfs_enable_region_ms);
    }
    if (min_mhz > max_mhz) {
        std::swap(min_mhz, max_mhz);
    }
    debug = env_bool("AMR_DVFS_DEBUG", debug);

    const char *interval_text = std::getenv("AMR_OMPT_EXEC_INTERVAL");
    if (interval_text) {
        output_exec_interval = std::max(1, std::atoi(interval_text));
    }

    const char *output_path = std::getenv("AMR_OMPT_OUT");
    output.open(output_path ? output_path : "ompt_dvfs/ompt_regions.csv");
    if (!output) {
        std::fprintf(stderr, "[AMR-DVFS] failed to open output CSV\n");
        return 0;
    }
    write_header();

    auto set_callback = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (!set_callback) {
        std::fprintf(stderr, "[AMR-DVFS] ompt_set_callback is unavailable\n");
        return 0;
    }

    set_callback(ompt_callback_parallel_begin,
                 reinterpret_cast<ompt_callback_t>(&on_parallel_begin));
    set_callback(ompt_callback_parallel_end,
                 reinterpret_cast<ompt_callback_t>(&on_parallel_end));
    set_callback(ompt_callback_implicit_task,
                 reinterpret_cast<ompt_callback_t>(&on_implicit_task));
    set_callback(ompt_callback_work,
                 reinterpret_cast<ompt_callback_t>(&on_work));

    msr_counter_enabled = can_open_current_msr_for_read();
    msr_control_enabled = can_open_current_msr_for_write();
    if (msr_counter_enabled || msr_control_enabled) {
        msr::msr_init();
    }
    if (msr_counter_enabled && reference_mhz <= 0.0) {
        reference_mhz = msr::platform_base_mhz_on_cpu(msr::current_cpu(), bus_mhz);
    }
    if (msr_control_enabled) {
        msr::reset_freq_all();
    } else {
        std::fprintf(stderr,
                     "[AMR-DVFS] MSR write unavailable; recording timing and planned targets only\n");
    }
    if (msr_counter_enabled && reference_mhz <= 0.0) {
        std::fprintf(stderr,
                     "[AMR-DVFS] MSR counters available, but base MHz is unknown; "
                     "set AMR_DVFS_BASE_MHZ to enable measured_mhz\n");
    }

    std::fprintf(stderr,
                 "[AMR-DVFS] enabled, writing %s every %d execution(s) per region, "
                 "target range %.0f-%.0f MHz, region DVFS enable >= %.3f ms, "
                 "disable <= %.3f ms, MSR control %s, counters %s, base %.0f MHz\n",
                 output_path ? output_path : "ompt_dvfs/ompt_regions.csv",
                 output_exec_interval,
                 min_mhz,
                 max_mhz,
                 dvfs_enable_region_ms,
                 dvfs_disable_region_ms,
                 msr_control_enabled ? "enabled" : "disabled",
                 msr_counter_enabled ? "enabled" : "disabled",
                 reference_mhz > 0.0 ? reference_mhz : 0.0);
    return 1;
}

void ompt_finalize(ompt_data_t *tool_data) {
    (void)tool_data;
    if (msr_control_enabled) {
        msr::reset_freq_all();
    }

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
