#include "resctrl_cat.hpp"

#include <algorithm>
#include <cerrno>
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

std::mutex cat_mutex;
std::string configured_root = "/sys/fs/resctrl";
std::string configured_prefix = "amr_l3";
bool cached_info_valid = false;
resctrl_cat::L3Info cached_info;
std::set<int> created_slots;
std::map<int, int> group_ways_by_slot;
std::map<int, std::uint64_t> group_mask_by_slot;
std::map<int, pid_t> assigned_tid_by_slot;

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

std::uint64_t read_hex_file(const std::string &path, std::uint64_t fallback) {
    std::string text;
    if (!read_file(path, &text)) {
        return fallback;
    }

    char *end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 16);
    return end != text.c_str() ? static_cast<std::uint64_t>(value) : fallback;
}

int popcount64(std::uint64_t value) {
    int count = 0;
    while (value != 0) {
        count += static_cast<int>(value & 1ULL);
        value >>= 1;
    }
    return count;
}

bool parse_hex_mask(const std::string &text, std::uint64_t *mask) {
    const std::string value = trim(text);
    if (value.empty()) {
        return false;
    }

    const char *start = value.c_str();
    if (value.size() > 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
        start += 2;
    }

    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(start, &end, 16);
    if (end == start) {
        return false;
    }
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
        ++end;
    }
    if (*end != '\0') {
        return false;
    }

    *mask = static_cast<std::uint64_t>(parsed);
    return true;
}

std::string hex_mask(std::uint64_t mask) {
    std::ostringstream out;
    out << std::hex << mask;
    return out.str();
}

std::vector<int> parse_l3_domains(const std::string &schemata) {
    std::vector<int> domains;
    std::istringstream lines(schemata);
    std::string line;
    while (std::getline(lines, line)) {
        line = trim(line);
        if (line.rfind("L3:", 0) != 0) {
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

std::uint64_t contiguous_mask_for_ways_locked(int requested_ways) {
    if (cached_info.cbm_mask == 0) {
        return 0;
    }

    const int min_ways = std::max(1, cached_info.min_cbm_bits);
    const int max_ways = std::max(min_ways, popcount64(cached_info.cbm_mask));
    int ways = requested_ways <= 0 ? min_ways : requested_ways;
    ways = std::clamp(ways, min_ways, max_ways);

    for (int start = 0; start < 64; ++start) {
        std::uint64_t mask = 0;
        int have = 0;
        for (int bit = start; bit < 64 && have < ways; ++bit) {
            const std::uint64_t bit_mask = 1ULL << bit;
            if ((cached_info.cbm_mask & bit_mask) == 0) {
                break;
            }
            mask |= bit_mask;
            ++have;
        }
        if (have == ways) {
            return mask;
        }
    }

    std::uint64_t mask = 0;
    int have = 0;
    for (int bit = 0; bit < 64 && have < ways; ++bit) {
        const std::uint64_t bit_mask = 1ULL << bit;
        if ((cached_info.cbm_mask & bit_mask) != 0) {
            mask |= bit_mask;
            ++have;
        }
    }
    return mask;
}

resctrl_cat::L3Info build_l3_info_locked(int fallback_min_cbm_bits) {
    resctrl_cat::L3Info info;
    info.min_cbm_bits = fallback_min_cbm_bits > 0 ? fallback_min_cbm_bits : 1;

    if (!is_directory(configured_root)) {
        return info;
    }

    const std::string l3_info_dir = join_path(join_path(configured_root, "info"), "L3");
    info.min_cbm_bits = read_int_file(join_path(l3_info_dir, "min_cbm_bits"),
                                      info.min_cbm_bits);
    if (info.min_cbm_bits <= 0) {
        info.min_cbm_bits = 1;
    }
    info.cbm_mask = read_hex_file(join_path(l3_info_dir, "cbm_mask"), 0);
    info.max_cbm_bits = popcount64(info.cbm_mask);

    std::string schemata;
    if (read_file(join_path(configured_root, "schemata"), &schemata)) {
        info.domains = parse_l3_domains(schemata);
    }

    info.available = info.cbm_mask != 0 && !info.domains.empty() &&
                     access(configured_root.c_str(), W_OK | X_OK) == 0 &&
                     access(join_path(configured_root, "tasks").c_str(), W_OK) == 0;
    return info;
}

std::string group_name_for_slot(int slot) {
    return configured_prefix + "_t" + std::to_string(std::max(0, slot));
}

std::string group_path_for_slot(int slot) {
    return join_path(configured_root, group_name_for_slot(slot));
}

std::string schemata_for_mask_locked(std::uint64_t mask) {
    std::ostringstream out;
    out << "L3:";
    for (std::size_t i = 0; i < cached_info.domains.size(); ++i) {
        if (i > 0) {
            out << ';';
        }
        out << cached_info.domains[i] << '=' << hex_mask(mask);
    }
    out << '\n';
    return out.str();
}

bool ensure_group_locked(int slot, int ways, std::uint64_t mask) {
    const std::string path = group_path_for_slot(slot);
    bool created = false;
    if (!is_directory(path)) {
        if (mkdir(path.c_str(), 0755) != 0) {
            return false;
        }
        created_slots.insert(slot);
        created = true;
    }

    const auto ways_iter = group_ways_by_slot.find(slot);
    const auto mask_iter = group_mask_by_slot.find(slot);
    if (!created &&
        ways_iter != group_ways_by_slot.end() && ways_iter->second == ways &&
        mask_iter != group_mask_by_slot.end() && mask_iter->second == mask) {
        return true;
    }

    if (!write_file(join_path(path, "schemata"), schemata_for_mask_locked(mask))) {
        return false;
    }
    group_ways_by_slot[slot] = ways;
    group_mask_by_slot[slot] = mask;
    return true;
}

bool write_tid_to_tasks(const std::string &tasks_path, pid_t tid) {
    if (tid <= 0) {
        return false;
    }
    return write_file(tasks_path, std::to_string(static_cast<long>(tid)) + "\n");
}

int current_cpu() {
    const int cpu = sched_getcpu();
    return cpu >= 0 ? cpu : -1;
}

pid_t current_tid() {
    const long tid = syscall(SYS_gettid);
    return tid > 0 ? static_cast<pid_t>(tid) : static_cast<pid_t>(-1);
}

}  // namespace

namespace resctrl_cat {

void configure(const char *root_path, const char *group_prefix) {
    std::lock_guard<std::mutex> lock(cat_mutex);
    if (root_path && root_path[0] != '\0') {
        configured_root = root_path;
    }
    if (group_prefix && group_prefix[0] != '\0') {
        configured_prefix = group_prefix;
    }
    cached_info_valid = false;
    cached_info = L3Info();
    created_slots.clear();
    group_ways_by_slot.clear();
    group_mask_by_slot.clear();
    assigned_tid_by_slot.clear();
}

L3Info l3_info(int fallback_min_cbm_bits) {
    std::lock_guard<std::mutex> lock(cat_mutex);
    cached_info = build_l3_info_locked(fallback_min_cbm_bits);
    cached_info_valid = true;
    return cached_info;
}

bool control_available() {
    std::lock_guard<std::mutex> lock(cat_mutex);
    if (!cached_info_valid) {
        cached_info = build_l3_info_locked(1);
        cached_info_valid = true;
    }
    return cached_info.available;
}

AssignmentResult assign_current_thread_l3(int slot, int requested_ways) {
    AssignmentResult result;
    result.cpu = current_cpu();
    result.tid = current_tid();

    std::lock_guard<std::mutex> lock(cat_mutex);
    if (!cached_info_valid) {
        cached_info = build_l3_info_locked(1);
        cached_info_valid = true;
    }
    if (!cached_info.available || result.tid <= 0) {
        return result;
    }

    const int ways = effective_ways(requested_ways);
    const std::uint64_t mask = contiguous_mask_for_ways_locked(ways);
    result.ways = ways;
    result.mask = hex_mask(mask);
    if (mask == 0 || !ensure_group_locked(slot, ways, mask)) {
        return result;
    }

    const auto assigned = assigned_tid_by_slot.find(slot);
    if (assigned != assigned_tid_by_slot.end() && assigned->second == result.tid) {
        result.applied = true;
        return result;
    }

    result.applied = write_tid_to_tasks(join_path(group_path_for_slot(slot), "tasks"),
                                        result.tid);
    if (result.applied) {
        assigned_tid_by_slot[slot] = result.tid;
    }
    return result;
}

AssignmentResult assign_current_thread_l3_mask(int slot, const std::string &requested_mask) {
    AssignmentResult result;
    result.cpu = current_cpu();
    result.tid = current_tid();

    std::lock_guard<std::mutex> lock(cat_mutex);
    if (!cached_info_valid) {
        cached_info = build_l3_info_locked(1);
        cached_info_valid = true;
    }
    if (!cached_info.available || result.tid <= 0) {
        return result;
    }

    std::uint64_t mask = 0;
    if (!parse_hex_mask(requested_mask, &mask)) {
        return result;
    }
    mask &= cached_info.cbm_mask;
    const int ways = popcount64(mask);
    if (mask == 0 || ways < std::max(1, cached_info.min_cbm_bits)) {
        return result;
    }

    result.ways = ways;
    result.mask = hex_mask(mask);
    if (!ensure_group_locked(slot, ways, mask)) {
        return result;
    }

    const auto assigned = assigned_tid_by_slot.find(slot);
    if (assigned != assigned_tid_by_slot.end() && assigned->second == result.tid) {
        result.applied = true;
        return result;
    }

    result.applied = write_tid_to_tasks(join_path(group_path_for_slot(slot), "tasks"),
                                        result.tid);
    if (result.applied) {
        assigned_tid_by_slot[slot] = result.tid;
    }
    return result;
}

bool release_task(pid_t tid) {
    std::lock_guard<std::mutex> lock(cat_mutex);
    if (tid <= 0 || !is_directory(configured_root)) {
        return false;
    }
    return write_tid_to_tasks(join_path(configured_root, "tasks"), tid);
}

bool cleanup_created_groups() {
    std::lock_guard<std::mutex> lock(cat_mutex);
    bool ok = true;
    for (const int slot : created_slots) {
        const std::string path = group_path_for_slot(slot);
        if (rmdir(path.c_str()) != 0 && errno != ENOENT) {
            ok = false;
        }
    }
    created_slots.clear();
    group_ways_by_slot.clear();
    group_mask_by_slot.clear();
    assigned_tid_by_slot.clear();
    return ok;
}

std::string root_path() {
    std::lock_guard<std::mutex> lock(cat_mutex);
    return configured_root;
}

std::string group_prefix() {
    std::lock_guard<std::mutex> lock(cat_mutex);
    return configured_prefix;
}

std::string mask_for_ways_text(int requested_ways) {
    std::lock_guard<std::mutex> lock(cat_mutex);
    if (!cached_info_valid) {
        cached_info = build_l3_info_locked(1);
        cached_info_valid = true;
    }
    return hex_mask(contiguous_mask_for_ways_locked(requested_ways));
}

int effective_ways(int requested_ways) {
    if (!cached_info_valid) {
        cached_info = build_l3_info_locked(1);
        cached_info_valid = true;
    }
    const int min_ways = std::max(1, cached_info.min_cbm_bits);
    const int max_ways = std::max(min_ways, popcount64(cached_info.cbm_mask));
    if (requested_ways <= 0) {
        return min_ways;
    }
    return std::clamp(requested_ways, min_ways, max_ways);
}

}  // namespace resctrl_cat
