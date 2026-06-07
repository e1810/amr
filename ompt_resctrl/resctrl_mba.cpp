#include "resctrl_mba.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
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
std::set<int> created_slots;
std::map<int, int> group_percent_by_slot;

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

int read_int_file(const std::string &path, int fallback) {
    std::string text;
    if (!read_file(path, &text)) {
        return fallback;
    }

    char *end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    return end != text.c_str() ? static_cast<int>(value) : fallback;
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

resctrl::MbaInfo build_mba_info_locked(int fallback_min_percent,
                                       int fallback_granularity_percent,
                                       int max_percent) {
    resctrl::MbaInfo info;
    info.min_percent = fallback_min_percent;
    info.granularity_percent = fallback_granularity_percent;
    info.max_percent = max_percent;

    if (!is_directory(configured_root)) {
        return info;
    }

    const std::string mb_info_dir = join_path(join_path(configured_root, "info"), "MB");
    info.min_percent = read_int_file(join_path(mb_info_dir, "min_bandwidth"),
                                     fallback_min_percent);
    info.granularity_percent = read_int_file(join_path(mb_info_dir, "bandwidth_gran"),
                                             fallback_granularity_percent);
    if (info.min_percent <= 0) {
        info.min_percent = fallback_min_percent;
    }
    if (info.granularity_percent <= 0) {
        info.granularity_percent = fallback_granularity_percent;
    }
    if (info.max_percent <= 0) {
        info.max_percent = 100;
    }
    if (info.min_percent > info.max_percent) {
        info.min_percent = info.max_percent;
    }

    std::string schemata;
    if (read_file(join_path(configured_root, "schemata"), &schemata)) {
        info.domains = parse_mb_domains(schemata);
    }

    info.available = !info.domains.empty() &&
                     access(configured_root.c_str(), W_OK | X_OK) == 0 &&
                     access(join_path(configured_root, "tasks").c_str(), W_OK) == 0;
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

std::string group_name_for_slot(int slot) {
    return configured_prefix + "_t" + std::to_string(std::max(0, slot));
}

std::string group_path_for_slot(int slot) {
    return join_path(configured_root, group_name_for_slot(slot));
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

bool ensure_group_locked(int slot, int percent) {
    const std::string path = group_path_for_slot(slot);
    bool created = false;
    if (!is_directory(path)) {
        if (mkdir(path.c_str(), 0755) != 0) {
            return false;
        }
        created_slots.insert(slot);
        created = true;
    }

    const auto iter = group_percent_by_slot.find(slot);
    if (!created && iter != group_percent_by_slot.end() && iter->second == percent) {
        return true;
    }

    if (!write_file(join_path(path, "schemata"), schemata_for_percent_locked(percent))) {
        return false;
    }
    group_percent_by_slot[slot] = percent;
    return true;
}

bool write_tid_to_tasks(const std::string &tasks_path, pid_t tid) {
    if (tid <= 0) {
        return false;
    }
    return write_file(tasks_path, std::to_string(static_cast<long>(tid)) + "\n");
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
    created_slots.clear();
    group_percent_by_slot.clear();
}

MbaInfo mba_info(int fallback_min_percent,
                 int fallback_granularity_percent,
                 int max_percent) {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    cached_info = build_mba_info_locked(fallback_min_percent,
                                        fallback_granularity_percent,
                                        max_percent);
    cached_info_valid = true;
    return cached_info;
}

bool control_available() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (!cached_info_valid) {
        cached_info = build_mba_info_locked(10, 10, 100);
        cached_info_valid = true;
    }
    return cached_info.available;
}

int current_cpu() {
    const int cpu = sched_getcpu();
    return cpu >= 0 ? cpu : -1;
}

pid_t current_tid() {
    const long tid = syscall(SYS_gettid);
    return tid > 0 ? static_cast<pid_t>(tid) : static_cast<pid_t>(-1);
}

AssignmentResult assign_current_thread_mba(int slot, double percent) {
    AssignmentResult result;
    result.cpu = current_cpu();
    result.tid = current_tid();

    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (!cached_info_valid) {
        cached_info = build_mba_info_locked(10, 10, 100);
        cached_info_valid = true;
    }
    if (!cached_info.available || result.tid <= 0) {
        return result;
    }

    const int normalized = normalize_percent_locked(percent);
    if (!ensure_group_locked(slot, normalized)) {
        return result;
    }

    result.applied = write_tid_to_tasks(join_path(group_path_for_slot(slot), "tasks"),
                                        result.tid);
    return result;
}

bool release_task(pid_t tid) {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    if (tid <= 0 || !is_directory(configured_root)) {
        return false;
    }
    return write_tid_to_tasks(join_path(configured_root, "tasks"), tid);
}

bool cleanup_created_groups() {
    std::lock_guard<std::mutex> lock(resctrl_mutex);
    bool ok = true;
    for (const int slot : created_slots) {
        const std::string path = group_path_for_slot(slot);
        if (rmdir(path.c_str()) != 0 && errno != ENOENT) {
            ok = false;
        }
    }
    created_slots.clear();
    group_percent_by_slot.clear();
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
