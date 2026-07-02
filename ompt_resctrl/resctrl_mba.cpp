#include "resctrl_mba.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sched.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

std::mutex resctrl_mutex;
std::string configured_root = "/sys/fs/resctrl";
std::string configured_prefix = "amr_mba";
bool cached_info_valid = false;
resctrl::MbaInfo cached_info;
std::set<std::string> created_groups;
std::map<std::string, int> group_percent_by_name;
std::map<pid_t, std::string> assigned_group_by_tid;
std::vector<std::string> mba_level_group_names;
std::string last_error_text;

std::string join_path(const std::string &a, const std::string &b) {
    if (a.empty()) {
        return b;
    }
    if (a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

bool is_directory(const std::string &path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string errno_diagnostic(const std::string &operation, const std::string &path) {
    return operation + " " + path + ": " + std::strerror(errno);
}

bool read_file(const std::string &path, std::string *out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    *out = buffer.str();
    return true;
}

bool write_file(const std::string &path, const std::string &text) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << text;
    return static_cast<bool>(out);
}

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool read_int_file(const std::string &path, int *out) {
    std::string text;
    if (!out || !read_file(path, &text)) {
        return false;
    }

    char *end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str()) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

std::vector<int> parse_mb_domains(const std::string &schemata) {
    std::vector<int> domains;
    std::istringstream lines(schemata);
    std::string line;
    while (std::getline(lines, line)) {
        line = trim(line);
        if (line.rfind("MB:", 0) != 0) {
            continue;
        }

        std::string entries = line.substr(3);
        std::istringstream parts(entries);
        std::string part;
        while (std::getline(parts, part, ';')) {
            part = trim(part);
            if (part.empty()) {
                continue;
            }

            const auto eq = part.find('=');
            if (eq == std::string::npos) {
                continue;
            }

            const std::string domain_text = part.substr(0, eq);
            char *end = nullptr;
            const long domain = std::strtol(domain_text.c_str(), &end, 10);
            if (end != domain_text.c_str()) {
                domains.push_back(static_cast<int>(domain));
            }
        }
        break;
    }

    std::sort(domains.begin(), domains.end());
    domains.erase(std::unique(domains.begin(), domains.end()), domains.end());
    return domains;
}

resctrl::MbaInfo build_mba_info_locked() {
    resctrl::MbaInfo info;
    info.min_percent = resctrl::kMbaLevelMinPercent;
    info.granularity_percent = resctrl::kMbaLevelStepPercent;
    info.max_percent = resctrl::kMbaLevelMaxPercent;

    if (!is_directory(configured_root)) {
        info.diagnostic = "resctrl root is missing or not a directory: " + configured_root;
        return info;
    }

    const std::string mb_info_dir = join_path(join_path(configured_root, "info"), "MB");
    const std::string min_path = join_path(mb_info_dir, "min_bandwidth");
    if (!read_int_file(min_path, &info.min_percent) || info.min_percent <= 0) {
        info.diagnostic = "cannot read positive MBA min_bandwidth: " + min_path;
        return info;
    }
    const std::string granularity_path = join_path(mb_info_dir, "bandwidth_gran");
    if (!read_int_file(granularity_path, &info.granularity_percent) ||
        info.granularity_percent <= 0) {
        info.diagnostic = "cannot read positive MBA bandwidth_gran: " + granularity_path;
        return info;
    }
    if (info.min_percent > info.max_percent) {
        info.diagnostic = "MBA min_bandwidth exceeds configured max percent";
        return info;
    }

    const std::string schemata_path = join_path(configured_root, "schemata");
    std::string schemata;
    if (!read_file(schemata_path, &schemata)) {
        info.diagnostic = "cannot read resctrl schemata: " + schemata_path;
        return info;
    }
    info.domains = parse_mb_domains(schemata);
    if (info.domains.empty()) {
        info.diagnostic = "MBA is not listed in resctrl schemata: " + schemata_path;
        return info;
    }

    const std::string tasks_path = join_path(configured_root, "tasks");
    if (access(configured_root.c_str(), W_OK | X_OK) != 0) {
        info.diagnostic = errno_diagnostic("resctrl root is not writable", configured_root);
        return info;
    }
    if (access(tasks_path.c_str(), W_OK) != 0) {
        info.diagnostic = errno_diagnostic("resctrl tasks is not writable", tasks_path);
        return info;
    }
    info.available = true;
    return info;
}

int normalize_percent_locked(double percent) {
    const int min_percent = cached_info.min_percent;
    const int max_percent = cached_info.max_percent;
    const int granularity = std::max(1, cached_info.granularity_percent);
    double value = std::clamp(percent,
                              static_cast<double>(min_percent),
                              static_cast<double>(max_percent));
    if (value >= static_cast<double>(max_percent)) {
        return max_percent;
    }

    const double steps = std::ceil((value - static_cast<double>(min_percent)) /
                                   static_cast<double>(granularity));
    const int rounded = min_percent + static_cast<int>(steps) * granularity;
    return std::clamp(rounded, min_percent, max_percent);
}


std::string group_name_for_mba_level(int level) {
    return configured_prefix + "_mba_level" +
           std::to_string(std::clamp(level, 0, resctrl::kMbaLevelCount - 1) + 1);
}

std::string group_path_for_name(const std::string &group_name) {
    return join_path(configured_root, group_name);
}

std::string schemata_for_percent_locked(int percent) {
    std::ostringstream out;
    out << "MB:";
    for (std::size_t i = 0; i < cached_info.domains.size(); ++i) {
        if (i > 0) {
            out << ';';
        }
        out << cached_info.domains[i] << '=' << percent;
    }
    out << '\n';
    return out.str();
}

bool ensure_named_group_locked(const std::string &group_name, int percent) {
    const std::string path = group_path_for_name(group_name);
    bool created = false;
    if (!is_directory(path)) {
        if (mkdir(path.c_str(), 0755) != 0) {
            last_error_text = errno_diagnostic("cannot create MBA group", path);
            return false;
        }
        created_groups.insert(group_name);
        created = true;
    }

    const auto iter = group_percent_by_name.find(group_name);
    if (!created && iter != group_percent_by_name.end() && iter->second == percent) {
        return true;
    }

    if (!write_file(join_path(path, "schemata"), schemata_for_percent_locked(percent))) {
        last_error_text = errno_diagnostic("cannot write MBA schemata", join_path(path, "schemata"));
        return false;
    }
    group_percent_by_name[group_name] = percent;
    return true;
}


bool write_tid_to_tasks(const std::string &tasks_path, pid_t tid) {
    if (tid <= 0) {
        last_error_text = "current thread ID is unavailable";
        return false;
    }
    if (!write_file(tasks_path, std::to_string(static_cast<long>(tid)) + "\n")) {
        last_error_text = errno_diagnostic("cannot assign TID " + std::to_string(static_cast<long>(tid)) +
                                           " to MBA group through", tasks_path);
        return false;
    }
    return true;
}

}  // namespace

namespace resctrl {

void configure(const char *root_path, const char *group_prefix) {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (root_path && root_path[0] != '\0') {
        configured_root = root_path;
    }
    if (group_prefix && group_prefix[0] != '\0') {
        configured_prefix = group_prefix;
    }
    cached_info_valid = false;
    cached_info = MbaInfo();
    created_groups.clear();
    group_percent_by_name.clear();
    assigned_group_by_tid.clear();
    mba_level_group_names.clear();
    last_error_text.clear();
}

MbaInfo mba_info() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    cached_info = build_mba_info_locked();
    cached_info_valid = true;
    last_error_text = cached_info.diagnostic;
    return cached_info;
}

bool control_available() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (!cached_info_valid) {
        cached_info = build_mba_info_locked();
        cached_info_valid = true;
    }
    last_error_text = cached_info.diagnostic;
    return cached_info.available;
}
std::string last_error() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    return last_error_text.empty() ? cached_info.diagnostic : last_error_text;
}


int current_cpu() {
    const int cpu = sched_getcpu();
    return cpu >= 0 ? cpu : -1;
}

pid_t current_tid() {
    const long tid = syscall(SYS_gettid);
    return tid > 0 ? static_cast<pid_t>(tid) : static_cast<pid_t>(-1);
}

int mba_percent_for_level(int level) {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (!cached_info_valid) {
        cached_info = build_mba_info_locked();
        cached_info_valid = true;
    }

    const int requested_percent =
        kMbaLevelPercents[static_cast<std::size_t>(
            std::clamp(level, 0, kMbaLevelCount - 1))];
    return normalize_percent_locked(requested_percent);
}

bool prepare_mba_level_groups() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (!cached_info_valid) {
        cached_info = build_mba_info_locked();
        cached_info_valid = true;
    }
    if (!cached_info.available) {
        last_error_text = cached_info.diagnostic;
        return false;
    }

    std::vector<std::string> level_names;
    level_names.reserve(kMbaLevelCount);
    for (int level = 0; level < kMbaLevelCount; ++level) {
        const int percent = normalize_percent_locked(
            kMbaLevelPercents[static_cast<std::size_t>(level)]);
        const std::string group_name = group_name_for_mba_level(level);
        if (!ensure_named_group_locked(group_name, percent)) {
            return false;
        }
        level_names.push_back(group_name);
    }

    mba_level_group_names = std::move(level_names);
    last_error_text.clear();
    return true;
}
AssignmentResult assign_current_thread_mba_level(int level) {
    AssignmentResult result;
    result.cpu = current_cpu();
    result.tid = current_tid();

    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (!cached_info_valid) {
        cached_info = build_mba_info_locked();
        cached_info_valid = true;
    }
    if (!cached_info.available) {
        last_error_text = cached_info.diagnostic;
        return result;
    }
    if (result.tid <= 0) {
        last_error_text = "current thread ID is unavailable";
        return result;
    }

    if (level < 0 || level >= static_cast<int>(mba_level_group_names.size())) {
        last_error_text = "requested MBA level was not prepared";
        return result;
    }

    result.group_name = mba_level_group_names[static_cast<std::size_t>(level)];
    const auto assigned = assigned_group_by_tid.find(result.tid);
    if (assigned != assigned_group_by_tid.end() && assigned->second == result.group_name) {
        result.applied = true;
        return result;
    }

    result.applied = write_tid_to_tasks(join_path(group_path_for_name(result.group_name), "tasks"),
                                        result.tid);
    if (result.applied) {
        assigned_group_by_tid[result.tid] = result.group_name;
    }
    return result;
}


bool release_task(pid_t tid) {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (tid <= 0 || !is_directory(configured_root)) {
        return false;
    }
    const bool released = write_tid_to_tasks(join_path(configured_root, "tasks"), tid);
    if (released) {
        assigned_group_by_tid.erase(tid);
    }
    return released;
}

bool cleanup_created_groups() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    bool ok = true;
    for (const auto &assignment : assigned_group_by_tid) {
        if (!write_tid_to_tasks(join_path(configured_root, "tasks"), assignment.first)) {
            ok = false;
        }
    }
    for (const std::string &group_name : created_groups) {
        const std::string path = group_path_for_name(group_name);
        if (rmdir(path.c_str()) != 0 && errno != ENOENT) {
            ok = false;
        }
    }
    created_groups.clear();
    group_percent_by_name.clear();
    assigned_group_by_tid.clear();
    mba_level_group_names.clear();
    return ok;
}

std::string root_path() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    return configured_root;
}

std::string group_prefix() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    return configured_prefix;
}

}  // namespace resctrl
