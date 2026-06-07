#include "resctrl_mon.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace {

std::mutex mon_mutex;
std::string configured_root = "/sys/fs/resctrl";
std::string configured_prefix = "amr_l3";

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

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool read_counter_file(const std::string &path, std::int64_t *value) {
    std::string text;
    if (!read_file(path, &text)) {
        return false;
    }

    text = trim(text);
    if (text.empty() || text == "Unavailable") {
        return false;
    }

    char *end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(text.c_str(), &end, 10);
    if (end == text.c_str() || errno == ERANGE) {
        return false;
    }

    *value = static_cast<std::int64_t>(parsed);
    return true;
}

std::vector<std::string> list_l3_monitor_dirs(const std::string &mon_data_path) {
    std::vector<std::string> dirs;
    DIR *dir = opendir(mon_data_path.c_str());
    if (!dir) {
        return dirs;
    }

    while (dirent *entry = readdir(dir)) {
        const std::string name(entry->d_name);
        if (name.rfind("mon_L3_", 0) != 0) {
            continue;
        }

        const std::string path = join_path(mon_data_path, name);
        if (is_directory(path)) {
            dirs.push_back(path);
        }
    }
    closedir(dir);

    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

void add_counter(const std::string &path, const char *filename, std::int64_t *sum) {
    std::int64_t value = -1;
    if (!read_counter_file(join_path(path, filename), &value)) {
        return;
    }

    if (*sum < 0) {
        *sum = 0;
    }
    *sum += value;
}

}  // namespace

namespace resctrl_mon {

void configure(const char *root_path, const char *group_prefix) {
    std::lock_guard<std::mutex> lock(mon_mutex);
    if (root_path && root_path[0] != '\0') {
        configured_root = root_path;
    }
    if (group_prefix && group_prefix[0] != '\0') {
        configured_prefix = group_prefix;
    }
}

Sample sample_slot(int slot) {
    std::string root;
    std::string prefix;
    {
        std::lock_guard<std::mutex> lock(mon_mutex);
        root = configured_root;
        prefix = configured_prefix;
    }

    const std::string group_path = join_path(root, prefix + "_t" + std::to_string(std::max(0, slot)));
    const std::string mon_data_path = join_path(group_path, "mon_data");

    Sample sample;
    const std::vector<std::string> domain_dirs = list_l3_monitor_dirs(mon_data_path);
    sample.domain_count = static_cast<int>(domain_dirs.size());
    for (const std::string &domain_dir : domain_dirs) {
        add_counter(domain_dir, "llc_occupancy", &sample.llc_occupancy_bytes);
        add_counter(domain_dir, "mbm_total_bytes", &sample.mbm_total_bytes);
        add_counter(domain_dir, "mbm_local_bytes", &sample.mbm_local_bytes);
        add_counter(domain_dir, "mbm_remote_bytes", &sample.mbm_remote_bytes);
    }

    sample.valid = sample.llc_occupancy_bytes >= 0 ||
                   sample.mbm_total_bytes >= 0 ||
                   sample.mbm_local_bytes >= 0 ||
                   sample.mbm_remote_bytes >= 0;
    return sample;
}

std::string root_path() {
    std::lock_guard<std::mutex> lock(mon_mutex);
    return configured_root;
}

std::string group_prefix() {
    std::lock_guard<std::mutex> lock(mon_mutex);
    return configured_prefix;
}

}  // namespace resctrl_mon
