#include <omp-tools.h>
#include <omp.h>

#include "resctrl_mba.hpp"
#include "resctrl_cat.hpp"
#include "resctrl_mon.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class ResourceKind {
    Mba,
    Cat,
};

struct RegionState {
    int id = 0;
    unsigned long long executions = 0;
    bool resctrl_enabled = false;
    double last_region_wall_ms = 0.0;
    std::vector<double> last_elapsed_ms;
    std::vector<double> last_target_mb_percent;
};

struct ParallelEvent {
    int region_id = 0;
    unsigned long long region_exec = 0;
    unsigned long long parallel_id = 0;
    const void *codeptr = nullptr;
    int requested_threads = 0;
    bool resctrl_enabled = false;
    double region_wall_ms = 0.0;
    std::atomic<int> team_size{0};
    std::vector<double> start_ms;
    std::vector<double> current_start_ms;
    std::vector<double> end_ms;
    std::vector<double> elapsed_ms;
    std::vector<double> target_mb_percent;
    std::vector<int> target_l3_ways;
    std::vector<std::string> target_l3_mask;
    std::vector<int> cpu_id;
    std::vector<pid_t> task_id;
    std::vector<unsigned char> has_started;
    std::vector<unsigned char> work_started;
    std::vector<unsigned char> resource_applied;
    bool monitor_enabled = false;
    std::vector<resctrl_mon::Sample> monitor_interval_start;
    std::vector<unsigned char> monitor_interval_active;
    std::vector<unsigned char> monitor_valid;
    std::vector<int> monitor_l3_domains;
    std::vector<std::int64_t> monitor_start_llc_occupancy_bytes;
    std::vector<std::int64_t> monitor_end_llc_occupancy_bytes;
    std::vector<std::int64_t> monitor_llc_occupancy_delta_bytes;
    std::vector<std::int64_t> monitor_start_mbm_total_bytes;
    std::vector<std::int64_t> monitor_end_mbm_total_bytes;
    std::vector<std::int64_t> monitor_mbm_total_delta_bytes;
    std::vector<std::int64_t> monitor_start_mbm_local_bytes;
    std::vector<std::int64_t> monitor_end_mbm_local_bytes;
    std::vector<std::int64_t> monitor_mbm_local_delta_bytes;
    std::vector<std::int64_t> monitor_start_mbm_remote_bytes;
    std::vector<std::int64_t> monitor_end_mbm_remote_bytes;
    std::vector<std::int64_t> monitor_mbm_remote_delta_bytes;
};

std::mutex state_mutex;
std::mutex output_mutex;
std::unordered_map<const void *, RegionState> region_by_codeptr;
std::atomic<unsigned long long> next_parallel_id{1};
Clock::time_point tool_start;
std::ofstream output;
int output_exec_interval = 1;
double max_mb_percent = 100.0;
double min_mb_percent = 10.0;
double mba_granularity_percent = 10.0;
double resctrl_enable_region_ms = 1.2;
double resctrl_disable_region_ms = 0.8;
double target_elapsed_fraction = 0.95;
double target_aggressiveness = 1.0;
ResourceKind resource_kind = ResourceKind::Mba;
std::string resource_kind_name = "mba";
int requested_l3_ways = 0;
int effective_l3_ways = 0;
std::string effective_l3_mask;
int unrestricted_l3_ways = 0;
std::string unrestricted_l3_mask;
std::vector<int> unrestricted_l3_threads;
std::string unrestricted_l3_threads_text = "none";
bool debug = false;
bool resctrl_control_enabled = false;
bool cat_control_enabled = false;
bool cat_apply_always = true;
bool cat_sticky_assignments = true;
bool cleanup_groups_on_finalize = false;
bool monitor_requested = false;
bool monitor_configured = false;

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

int env_int(const char *name, int fallback) {
    const char *text = std::getenv(name);
    if (!text) {
        return fallback;
    }

    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end != text ? static_cast<int>(value) : fallback;
}

int hex_mask_popcount(const std::string &text, std::uint64_t cbm_mask) {
    if (text.empty()) {
        return 0;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 16);
    if (end == text.c_str()) {
        return 0;
    }
    std::uint64_t mask = static_cast<std::uint64_t>(parsed);
    if (cbm_mask != 0) {
        mask &= cbm_mask;
    }
    int count = 0;
    while (mask != 0) {
        count += static_cast<int>(mask & 1ULL);
        mask >>= 1;
    }
    return count;
}

std::vector<int> env_thread_list(const char *name) {
    std::vector<int> result;
    const char *text = std::getenv(name);
    if (!text || text[0] == '\0') {
        return result;
    }

    const char *cursor = text;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '	' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }

        char *end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor) {
            while (*cursor != '\0' && *cursor != ',') {
                ++cursor;
            }
            continue;
        }

        if (value >= 0 &&
            std::find(result.begin(), result.end(), static_cast<int>(value)) == result.end()) {
            result.push_back(static_cast<int>(value));
        }
        cursor = end;
    }
    return result;
}

std::string thread_list_text(const std::vector<int> &threads) {
    if (threads.empty()) {
        return "none";
    }

    std::string text;
    for (std::size_t i = 0; i < threads.size(); ++i) {
        if (i > 0) {
            text += ',';
        }
        text += std::to_string(threads[i]);
    }
    return text;
}

bool is_unrestricted_l3_thread(std::size_t slot) {
    return std::find(unrestricted_l3_threads.begin(),
                     unrestricted_l3_threads.end(),
                     static_cast<int>(slot)) != unrestricted_l3_threads.end();
}

ResourceKind parse_resource_kind() {
    const char *text = std::getenv("AMR_RESCTRL_RESOURCE");
    if (!text || text[0] == '\0') {
        return ResourceKind::Mba;
    }

    const std::string value(text);
    if (value == "cat" || value == "l3") {
        return ResourceKind::Cat;
    }
    return ResourceKind::Mba;
}

const char *resource_kind_label(ResourceKind kind) {
    return kind == ResourceKind::Cat ? "cat" : "mba";
}

double normalize_mb_percent(double value) {
    if (!std::isfinite(value)) {
        return max_mb_percent;
    }

    value = std::clamp(value, min_mb_percent, max_mb_percent);
    if (value >= max_mb_percent) {
        return max_mb_percent;
    }
    if (mba_granularity_percent <= 0.0) {
        return value;
    }

    const double steps = std::ceil((value - min_mb_percent) / mba_granularity_percent);
    return std::clamp(min_mb_percent + steps * mba_granularity_percent,
                      min_mb_percent,
                      max_mb_percent);
}

void set_first_counter(std::vector<std::int64_t> *values,
                       std::size_t slot,
                       std::int64_t value) {
    if (value < 0 || !values || slot >= values->size()) {
        return;
    }
    if ((*values)[slot] < 0) {
        (*values)[slot] = value;
    }
}

void set_last_counter(std::vector<std::int64_t> *values,
                      std::size_t slot,
                      std::int64_t value) {
    if (value < 0 || !values || slot >= values->size()) {
        return;
    }
    (*values)[slot] = value;
}

void add_counter_delta(std::vector<std::int64_t> *values,
                       std::size_t slot,
                       std::int64_t begin,
                       std::int64_t end) {
    if (begin < 0 || end < 0 || !values || slot >= values->size()) {
        return;
    }
    if ((*values)[slot] < 0) {
        (*values)[slot] = 0;
    }
    (*values)[slot] += end - begin;
}

void record_monitor_begin(ParallelEvent *event, unsigned int thread_num) {
    if (!event || !event->monitor_enabled || !has_thread_slot(event, thread_num)) {
        return;
    }

    const std::size_t slot = static_cast<std::size_t>(thread_num);
    if (slot >= event->monitor_interval_start.size() ||
        slot >= event->monitor_interval_active.size() ||
        slot >= event->resource_applied.size() ||
        !event->resource_applied[slot]) {
        return;
    }

    const resctrl_mon::Sample sample = resctrl_mon::sample_slot(static_cast<int>(slot));
    if (!sample.valid) {
        event->monitor_interval_active[slot] = 0;
        return;
    }

    event->monitor_interval_start[slot] = sample;
    event->monitor_interval_active[slot] = 1;
    if (slot < event->monitor_valid.size()) {
        event->monitor_valid[slot] = 1;
    }
    if (slot < event->monitor_l3_domains.size()) {
        event->monitor_l3_domains[slot] = std::max(event->monitor_l3_domains[slot],
                                                   sample.domain_count);
    }

    set_first_counter(&event->monitor_start_llc_occupancy_bytes,
                      slot,
                      sample.llc_occupancy_bytes);
    set_first_counter(&event->monitor_start_mbm_total_bytes,
                      slot,
                      sample.mbm_total_bytes);
    set_first_counter(&event->monitor_start_mbm_local_bytes,
                      slot,
                      sample.mbm_local_bytes);
    set_first_counter(&event->monitor_start_mbm_remote_bytes,
                      slot,
                      sample.mbm_remote_bytes);
}

void record_monitor_end(ParallelEvent *event, unsigned int thread_num) {
    if (!event || !event->monitor_enabled || !has_thread_slot(event, thread_num)) {
        return;
    }

    const std::size_t slot = static_cast<std::size_t>(thread_num);
    if (slot >= event->monitor_interval_start.size() ||
        slot >= event->monitor_interval_active.size() ||
        !event->monitor_interval_active[slot]) {
        return;
    }

    event->monitor_interval_active[slot] = 0;
    const resctrl_mon::Sample begin = event->monitor_interval_start[slot];
    const resctrl_mon::Sample end = resctrl_mon::sample_slot(static_cast<int>(slot));
    if (!end.valid) {
        return;
    }

    if (slot < event->monitor_valid.size()) {
        event->monitor_valid[slot] = 1;
    }
    if (slot < event->monitor_l3_domains.size()) {
        event->monitor_l3_domains[slot] = std::max(event->monitor_l3_domains[slot],
                                                   end.domain_count);
    }

    set_last_counter(&event->monitor_end_llc_occupancy_bytes,
                     slot,
                     end.llc_occupancy_bytes);
    set_last_counter(&event->monitor_end_mbm_total_bytes,
                     slot,
                     end.mbm_total_bytes);
    set_last_counter(&event->monitor_end_mbm_local_bytes,
                     slot,
                     end.mbm_local_bytes);
    set_last_counter(&event->monitor_end_mbm_remote_bytes,
                     slot,
                     end.mbm_remote_bytes);

    add_counter_delta(&event->monitor_llc_occupancy_delta_bytes,
                      slot,
                      begin.llc_occupancy_bytes,
                      end.llc_occupancy_bytes);
    add_counter_delta(&event->monitor_mbm_total_delta_bytes,
                      slot,
                      begin.mbm_total_bytes,
                      end.mbm_total_bytes);
    add_counter_delta(&event->monitor_mbm_local_delta_bytes,
                      slot,
                      begin.mbm_local_bytes,
                      end.mbm_local_bytes);
    add_counter_delta(&event->monitor_mbm_remote_delta_bytes,
                      slot,
                      begin.mbm_remote_bytes,
                      end.mbm_remote_bytes);
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
    record_monitor_end(event, thread_num);
    event->work_started[slot] = 0;
}

std::vector<double> plan_targets(const RegionState &region, std::size_t slots) {
    std::vector<double> targets(slots, max_mb_percent);
    if (region.executions <= 1 || region.last_elapsed_ms.empty()) {
        return targets;
    }

    std::vector<double> work_estimates(slots, 0.0);
    std::size_t critical_slot = slots;
    double critical_elapsed_ms = 0.0;
    for (std::size_t i = 0; i < slots && i < region.last_elapsed_ms.size(); ++i) {
        const double previous_percent =
            i < region.last_target_mb_percent.size() && region.last_target_mb_percent[i] > 0.0
                ? region.last_target_mb_percent[i]
                : max_mb_percent;
        work_estimates[i] = region.last_elapsed_ms[i] * previous_percent;
        if (region.last_elapsed_ms[i] > critical_elapsed_ms) {
            critical_elapsed_ms = region.last_elapsed_ms[i];
            critical_slot = i;
        }
    }
    if (critical_slot >= slots || work_estimates[critical_slot] <= 0.0) {
        return targets;
    }

    const double target_work = work_estimates[critical_slot] * target_elapsed_fraction;
    for (std::size_t i = 0; i < slots && i < region.last_elapsed_ms.size(); ++i) {
        double ratio = work_estimates[i] / target_work;
        if (ratio < 1.0 && target_aggressiveness > 1.0) {
            ratio = std::pow(ratio, target_aggressiveness);
        }
        targets[i] = normalize_mb_percent(max_mb_percent * ratio);
    }
    return targets;
}

void write_header() {
    output << "parallel_id,region_id,region_exec,thread_id,team_size,"
           << "requested_threads,start_ms,end_ms,elapsed_ms,codeptr,"
           << "cpu_id,target_mb_percent,resource_applied,"
           << "resctrl_enabled,region_wall_ms,resctrl_enable_ms,resctrl_disable_ms,"
           << "resource_kind,target_l3_ways,target_l3_mask,"
           << "mon_valid,mon_l3_domains,"
           << "mon_start_llc_occupancy_bytes,mon_end_llc_occupancy_bytes,"
           << "mon_llc_occupancy_delta_bytes,"
           << "mon_start_mbm_total_bytes,mon_end_mbm_total_bytes,"
           << "mon_mbm_total_delta_bytes,"
           << "mon_start_mbm_local_bytes,mon_end_mbm_local_bytes,"
           << "mon_mbm_local_delta_bytes,"
           << "mon_start_mbm_remote_bytes,mon_end_mbm_remote_bytes,"
           << "mon_mbm_remote_delta_bytes\n";
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
    int cpu = resctrl::current_cpu();
    pid_t tid = resctrl::current_tid();
    bool applied = false;

    if (resource_kind == ResourceKind::Cat) {
        if (cat_control_enabled && (cat_apply_always || event->resctrl_enabled)) {
            const std::string target_mask =
                slot < event->target_l3_mask.size() ? event->target_l3_mask[slot] : std::string();
            const resctrl_cat::AssignmentResult assignment =
                target_mask.empty()
                    ? resctrl_cat::assign_current_thread_l3(
                          static_cast<int>(slot),
                          slot < event->target_l3_ways.size() ? event->target_l3_ways[slot] : requested_l3_ways)
                    : resctrl_cat::assign_current_thread_l3_mask(static_cast<int>(slot), target_mask);
            cpu = assignment.cpu;
            tid = assignment.tid;
            applied = assignment.applied;
            if (slot < event->target_l3_ways.size()) {
                event->target_l3_ways[slot] = assignment.ways;
            }
            if (slot < event->target_l3_mask.size()) {
                event->target_l3_mask[slot] = assignment.mask;
            }
        }
    } else if (event->resctrl_enabled && resctrl_control_enabled &&
               event->target_mb_percent[slot] < max_mb_percent) {
        const resctrl::AssignmentResult assignment = resctrl::assign_current_thread_mba(
            static_cast<int>(slot),
            event->target_mb_percent[slot]);
        cpu = assignment.cpu;
        tid = assignment.tid;
        applied = assignment.applied;
    }

    event->cpu_id[slot] = cpu;
    event->task_id[slot] = tid;
    event->resource_applied[slot] = applied ? 1 : 0;

    if (debug) {
        if (resource_kind == ResourceKind::Cat) {
            std::fprintf(stderr,
                         "[AMR-RESCTRL] region=%d exec=%llu thread=%u tid=%ld cpu=%d "
                         "target_l3=%dway mask=%s %s\n",
                         event->region_id,
                         event->region_exec,
                         thread_num,
                         static_cast<long>(tid),
                         cpu,
                         slot < event->target_l3_ways.size() ? event->target_l3_ways[slot] : 0,
                         slot < event->target_l3_mask.size() ? event->target_l3_mask[slot].c_str() : "",
                         applied ? "applied" : "planned");
        } else {
            std::fprintf(stderr,
                         "[AMR-RESCTRL] region=%d exec=%llu thread=%u tid=%ld cpu=%d "
                         "target=%.0f%% %s\n",
                         event->region_id,
                         event->region_exec,
                         thread_num,
                         static_cast<long>(tid),
                         cpu,
                         event->target_mb_percent[slot],
                         applied ? "applied" : "planned");
        }
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
    if (resource_kind == ResourceKind::Cat && cat_sticky_assignments) {
        return;
    }

    std::vector<pid_t> reset_tasks;
    for (int tid = 0; tid < team_size; ++tid) {
        const std::size_t index = static_cast<std::size_t>(tid);
        if (index >= event.task_id.size() || index >= event.resource_applied.size() ||
            !event.resource_applied[index] || event.task_id[index] <= 0) {
            continue;
        }

        const pid_t task = event.task_id[index];
        if (std::find(reset_tasks.begin(), reset_tasks.end(), task) == reset_tasks.end()) {
            reset_tasks.push_back(task);
            if (resource_kind == ResourceKind::Cat) {
                resctrl_cat::release_task(task);
            } else {
                resctrl::release_task(task);
            }
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
    region.last_target_mb_percent.assign(static_cast<std::size_t>(team_size), max_mb_percent);
    for (int tid = 0; tid < team_size; ++tid) {
        const std::size_t index = static_cast<std::size_t>(tid);
        region.last_elapsed_ms[index] =
            index < event.elapsed_ms.size() ? event.elapsed_ms[index] : 0.0;
        region.last_target_mb_percent[index] =
            index < event.target_mb_percent.size() ? event.target_mb_percent[index] : max_mb_percent;
    }

    region.last_region_wall_ms = event.region_wall_ms;
    if (!region.resctrl_enabled && event.region_wall_ms >= resctrl_enable_region_ms) {
        region.resctrl_enabled = true;
    } else if (region.resctrl_enabled && event.region_wall_ms <= resctrl_disable_region_ms) {
        region.resctrl_enabled = false;
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
        const double target =
            index < event.target_mb_percent.size() ? event.target_mb_percent[index] : max_mb_percent;
        const int applied =
            index < event.resource_applied.size() ? static_cast<int>(event.resource_applied[index]) : 0;
        const int l3_ways =
            index < event.target_l3_ways.size() ? event.target_l3_ways[index] : 0;
        const std::string l3_mask =
            index < event.target_l3_mask.size() ? event.target_l3_mask[index] : "";
        const int mon_valid =
            index < event.monitor_valid.size() ? static_cast<int>(event.monitor_valid[index]) : 0;
        const int mon_domains =
            index < event.monitor_l3_domains.size() ? event.monitor_l3_domains[index] : 0;
        const std::int64_t mon_start_llc =
            index < event.monitor_start_llc_occupancy_bytes.size()
                ? event.monitor_start_llc_occupancy_bytes[index]
                : -1;
        const std::int64_t mon_end_llc =
            index < event.monitor_end_llc_occupancy_bytes.size()
                ? event.monitor_end_llc_occupancy_bytes[index]
                : -1;
        const std::int64_t mon_delta_llc =
            index < event.monitor_llc_occupancy_delta_bytes.size()
                ? event.monitor_llc_occupancy_delta_bytes[index]
                : -1;
        const std::int64_t mon_start_total =
            index < event.monitor_start_mbm_total_bytes.size()
                ? event.monitor_start_mbm_total_bytes[index]
                : -1;
        const std::int64_t mon_end_total =
            index < event.monitor_end_mbm_total_bytes.size()
                ? event.monitor_end_mbm_total_bytes[index]
                : -1;
        const std::int64_t mon_delta_total =
            index < event.monitor_mbm_total_delta_bytes.size()
                ? event.monitor_mbm_total_delta_bytes[index]
                : -1;
        const std::int64_t mon_start_local =
            index < event.monitor_start_mbm_local_bytes.size()
                ? event.monitor_start_mbm_local_bytes[index]
                : -1;
        const std::int64_t mon_end_local =
            index < event.monitor_end_mbm_local_bytes.size()
                ? event.monitor_end_mbm_local_bytes[index]
                : -1;
        const std::int64_t mon_delta_local =
            index < event.monitor_mbm_local_delta_bytes.size()
                ? event.monitor_mbm_local_delta_bytes[index]
                : -1;
        const std::int64_t mon_start_remote =
            index < event.monitor_start_mbm_remote_bytes.size()
                ? event.monitor_start_mbm_remote_bytes[index]
                : -1;
        const std::int64_t mon_end_remote =
            index < event.monitor_end_mbm_remote_bytes.size()
                ? event.monitor_end_mbm_remote_bytes[index]
                : -1;
        const std::int64_t mon_delta_remote =
            index < event.monitor_mbm_remote_delta_bytes.size()
                ? event.monitor_mbm_remote_delta_bytes[index]
                : -1;

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
               << applied << ','
               << (event.resctrl_enabled ? 1 : 0) << ','
               << event.region_wall_ms << ','
               << resctrl_enable_region_ms << ','
               << resctrl_disable_region_ms << ','
               << resource_kind_label(resource_kind) << ','
               << l3_ways << ','
               << l3_mask << ','
               << mon_valid << ','
               << mon_domains << ','
               << mon_start_llc << ','
               << mon_end_llc << ','
               << mon_delta_llc << ','
               << mon_start_total << ','
               << mon_end_total << ','
               << mon_delta_total << ','
               << mon_start_local << ','
               << mon_end_local << ','
               << mon_delta_local << ','
               << mon_start_remote << ','
               << mon_end_remote << ','
               << mon_delta_remote << '\n';
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
    event->target_l3_ways.assign(slots, effective_l3_ways);
    event->target_l3_mask.assign(slots, effective_l3_mask);
    if (resource_kind == ResourceKind::Cat) {
        for (std::size_t slot = 0; slot < static_cast<std::size_t>(slots); ++slot) {
            if (is_unrestricted_l3_thread(slot)) {
                event->target_l3_ways[slot] = unrestricted_l3_ways;
                event->target_l3_mask[slot] = unrestricted_l3_mask;
            }
        }
    }
    event->cpu_id.assign(slots, -1);
    event->task_id.assign(slots, static_cast<pid_t>(-1));
    event->has_started.assign(slots, 0);
    event->work_started.assign(slots, 0);
    event->resource_applied.assign(slots, 0);
    event->monitor_interval_start.assign(slots, resctrl_mon::Sample());
    event->monitor_interval_active.assign(slots, 0);
    event->monitor_valid.assign(slots, 0);
    event->monitor_l3_domains.assign(slots, 0);
    event->monitor_start_llc_occupancy_bytes.assign(slots, -1);
    event->monitor_end_llc_occupancy_bytes.assign(slots, -1);
    event->monitor_llc_occupancy_delta_bytes.assign(slots, -1);
    event->monitor_start_mbm_total_bytes.assign(slots, -1);
    event->monitor_end_mbm_total_bytes.assign(slots, -1);
    event->monitor_mbm_total_delta_bytes.assign(slots, -1);
    event->monitor_start_mbm_local_bytes.assign(slots, -1);
    event->monitor_end_mbm_local_bytes.assign(slots, -1);
    event->monitor_mbm_local_delta_bytes.assign(slots, -1);
    event->monitor_start_mbm_remote_bytes.assign(slots, -1);
    event->monitor_end_mbm_remote_bytes.assign(slots, -1);
    event->monitor_mbm_remote_delta_bytes.assign(slots, -1);

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
        event->resctrl_enabled = state.resctrl_enabled;
        if (resource_kind == ResourceKind::Mba) {
            event->target_mb_percent =
                event->resctrl_enabled
                    ? plan_targets(state, static_cast<std::size_t>(slots))
                    : std::vector<double>(static_cast<std::size_t>(slots), max_mb_percent);
        } else {
            event->resctrl_enabled = cat_apply_always || state.resctrl_enabled;
            event->target_mb_percent =
                std::vector<double>(static_cast<std::size_t>(slots), max_mb_percent);
        }
    }

    event->monitor_enabled = monitor_configured && should_write_execution(*event);
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
        record_monitor_begin(event, thread_num);
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
    resource_kind = parse_resource_kind();
    resource_kind_name = resource_kind_label(resource_kind);
    requested_l3_ways = env_int("AMR_RESCTRL_CAT_L3_WAYS", requested_l3_ways);
    cat_apply_always = env_bool("AMR_RESCTRL_CAT_ALWAYS", cat_apply_always);
    cat_sticky_assignments = env_bool("AMR_RESCTRL_CAT_STICKY", cat_sticky_assignments);
    max_mb_percent = env_double("AMR_RESCTRL_MAX_MB_PERCENT",
                                env_double("AMR_RESCTRL_MAX_PERCENT", max_mb_percent));
    min_mb_percent = env_double("AMR_RESCTRL_MIN_MB_PERCENT",
                                env_double("AMR_RESCTRL_MIN_PERCENT", min_mb_percent));
    mba_granularity_percent = env_double("AMR_RESCTRL_MB_GRANULARITY_PERCENT",
                                         mba_granularity_percent);
    resctrl_enable_region_ms =
        env_double("AMR_RESCTRL_ENABLE_REGION_MS", resctrl_enable_region_ms);
    resctrl_disable_region_ms =
        env_double("AMR_RESCTRL_DISABLE_REGION_MS", resctrl_disable_region_ms);
    target_elapsed_fraction =
        env_double("AMR_RESCTRL_TARGET_ELAPSED_FRACTION", target_elapsed_fraction);
    target_aggressiveness =
        env_double("AMR_RESCTRL_AGGRESSIVENESS", target_aggressiveness);
    if (resctrl_disable_region_ms > resctrl_enable_region_ms) {
        std::swap(resctrl_disable_region_ms, resctrl_enable_region_ms);
    }
    if (min_mb_percent > max_mb_percent) {
        std::swap(min_mb_percent, max_mb_percent);
    }
    if (target_elapsed_fraction <= 0.0) {
        target_elapsed_fraction = 0.95;
    }
    if (target_aggressiveness < 1.0) {
        target_aggressiveness = 1.0;
    }
    debug = env_bool("AMR_RESCTRL_DEBUG", debug);
    cleanup_groups_on_finalize = env_bool("AMR_RESCTRL_CLEANUP_GROUPS", false);
    monitor_requested = env_bool("AMR_RESCTRL_MONITOR", monitor_requested);

    if (resource_kind == ResourceKind::Cat) {
        resctrl_cat::configure(std::getenv("AMR_RESCTRL_ROOT"),
                               std::getenv("AMR_RESCTRL_CAT_GROUP_PREFIX"));
        const resctrl_cat::L3Info cat_info = resctrl_cat::l3_info(1);
        cat_control_enabled = cat_info.available;
        effective_l3_ways = resctrl_cat::effective_ways(requested_l3_ways);
        effective_l3_mask = resctrl_cat::mask_for_ways_text(requested_l3_ways);
        unrestricted_l3_ways = cat_info.max_cbm_bits > 0 ? cat_info.max_cbm_bits : effective_l3_ways;
        unrestricted_l3_mask = resctrl_cat::mask_for_ways_text(unrestricted_l3_ways);
        const char *unrestricted_mask_env = std::getenv("AMR_RESCTRL_CAT_UNRESTRICTED_L3_MASK");
        if (unrestricted_mask_env && unrestricted_mask_env[0] != '\0') {
            unrestricted_l3_mask = unrestricted_mask_env;
            const int mask_ways = hex_mask_popcount(unrestricted_l3_mask, cat_info.cbm_mask);
            if (mask_ways > 0) {
                unrestricted_l3_ways = mask_ways;
            }
        }
        unrestricted_l3_threads = env_thread_list("AMR_RESCTRL_CAT_UNRESTRICTED_THREADS");
        unrestricted_l3_threads_text = thread_list_text(unrestricted_l3_threads);
    } else {
        resctrl::configure(std::getenv("AMR_RESCTRL_ROOT"),
                           std::getenv("AMR_RESCTRL_GROUP_PREFIX"));
        const resctrl::MbaInfo mba_info =
            resctrl::mba_info(static_cast<int>(std::round(min_mb_percent)),
                              static_cast<int>(std::round(mba_granularity_percent)),
                              static_cast<int>(std::round(max_mb_percent)));
        min_mb_percent = env_double("AMR_RESCTRL_MIN_MB_PERCENT",
                                    env_double("AMR_RESCTRL_MIN_PERCENT",
                                               static_cast<double>(mba_info.min_percent)));
        mba_granularity_percent =
            env_double("AMR_RESCTRL_MB_GRANULARITY_PERCENT",
                       static_cast<double>(mba_info.granularity_percent));
        if (min_mb_percent > max_mb_percent) {
            std::swap(min_mb_percent, max_mb_percent);
        }
        resctrl_control_enabled = mba_info.available;
    }

    if (resource_kind == ResourceKind::Cat) {
        const std::string root = resctrl_cat::root_path();
        const std::string prefix = resctrl_cat::group_prefix();
        resctrl_mon::configure(root.c_str(), prefix.c_str());
        monitor_configured = monitor_requested && cat_control_enabled;
    } else {
        const std::string root = resctrl::root_path();
        const std::string prefix = resctrl::group_prefix();
        resctrl_mon::configure(root.c_str(), prefix.c_str());
        monitor_configured = monitor_requested && resctrl_control_enabled;
    }

    const char *interval_text = std::getenv("AMR_OMPT_EXEC_INTERVAL");
    if (interval_text) {
        output_exec_interval = std::max(1, std::atoi(interval_text));
    }

    const char *output_path = std::getenv("AMR_OMPT_OUT");
    output.open(output_path ? output_path : "ompt_resctrl/ompt_regions.csv");
    if (!output) {
        std::fprintf(stderr, "[AMR-RESCTRL] failed to open output CSV\n");
        return 0;
    }
    write_header();

    auto set_callback = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (!set_callback) {
        std::fprintf(stderr, "[AMR-RESCTRL] ompt_set_callback is unavailable\n");
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

    if (resource_kind == ResourceKind::Cat && !cat_control_enabled) {
        std::fprintf(stderr,
                     "[AMR-RESCTRL] resctrl L3 CAT write unavailable; recording timing only\n");
    } else if (resource_kind == ResourceKind::Mba && !resctrl_control_enabled) {
        std::fprintf(stderr,
                     "[AMR-RESCTRL] resctrl MBA write unavailable; recording timing and "
                     "planned targets only\n");
    }

    if (resource_kind == ResourceKind::Cat) {
        std::fprintf(stderr,
                     "[AMR-RESCTRL] enabled, writing %s every %d execution(s) per region, "
                     "resource cat, requested L3 ways %d, effective L3 ways %d, "
                     "mask %s, unrestricted threads %s use L3 ways %d mask %s, "
                     "cat always %s, sticky %s, control %s, monitor %s, "
                     "cleanup %s, root %s, group prefix %s\n",
                     output_path ? output_path : "ompt_resctrl/ompt_regions.csv",
                     output_exec_interval,
                     requested_l3_ways,
                     effective_l3_ways,
                     effective_l3_mask.c_str(),
                     unrestricted_l3_threads_text.c_str(),
                     unrestricted_l3_ways,
                     unrestricted_l3_mask.c_str(),
                     cat_apply_always ? "enabled" : "disabled",
                     cat_sticky_assignments ? "enabled" : "disabled",
                     cat_control_enabled ? "enabled" : "disabled",
                     monitor_configured ? "enabled" : "disabled",
                     cleanup_groups_on_finalize ? "enabled" : "disabled",
                     resctrl_cat::root_path().c_str(),
                     resctrl_cat::group_prefix().c_str());
    } else {
        std::fprintf(stderr,
                     "[AMR-RESCTRL] enabled, writing %s every %d execution(s) per region, "
                     "resource mba, MBA target range %.0f-%.0f%%, granularity %.0f%%, "
                     "target fraction %.2f, aggressiveness %.2f, region resctrl enable "
                     ">= %.3f ms, disable <= %.3f ms, control %s, monitor %s, "
                     "cleanup %s, root %s, group prefix %s\n",
                     output_path ? output_path : "ompt_resctrl/ompt_regions.csv",
                     output_exec_interval,
                     min_mb_percent,
                     max_mb_percent,
                     mba_granularity_percent,
                     target_elapsed_fraction,
                     target_aggressiveness,
                     resctrl_enable_region_ms,
                     resctrl_disable_region_ms,
                     resctrl_control_enabled ? "enabled" : "disabled",
                     monitor_configured ? "enabled" : "disabled",
                     cleanup_groups_on_finalize ? "enabled" : "disabled",
                     resctrl::root_path().c_str(),
                     resctrl::group_prefix().c_str());
    }
    return 1;
}

void ompt_finalize(ompt_data_t *tool_data) {
    (void)tool_data;
    if (cleanup_groups_on_finalize) {
        if (resource_kind == ResourceKind::Cat && cat_control_enabled) {
            resctrl_cat::cleanup_created_groups();
        } else if (resource_kind == ResourceKind::Mba && resctrl_control_enabled) {
            resctrl::cleanup_created_groups();
        }
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
