#ifndef RESCTRL_MBA_HPP
#define RESCTRL_MBA_HPP

#include <sys/types.h>

#include <array>
#include <string>
#include <vector>

namespace resctrl {
constexpr int kMbaLevelCount = 7;
constexpr int kMbaLevelMinPercent = 20;
constexpr int kMbaLevelStepPercent = 10;
constexpr int kMbaLevelMaxPercent = 100;
constexpr std::array<int, kMbaLevelCount> kMbaLevelPercents = {
    20, 30, 40, 50, 60, 80, 100,
};

struct MbaInfo {
    bool available = false;
    std::string diagnostic;
    int min_percent = 10;
    int granularity_percent = 10;
    int max_percent = 100;
    std::vector<int> domains;
};

struct AssignmentResult {
    bool applied = false;
    int cpu = -1;
    pid_t tid = -1;
    std::string group_name;
};

void configure(const char *root_path, const char *group_prefix);
MbaInfo mba_info();
bool control_available();
std::string last_error();

int current_cpu();
pid_t current_tid();

// Levels are fixed to the seven MBA groups supported by the target hardware.
// The level groups must be prepared once during tool initialization before tasks
// are assigned to them.
int mba_percent_for_level(int level);
bool prepare_mba_level_groups();
AssignmentResult assign_current_thread_mba_level(int level);

bool release_task(pid_t tid);
bool cleanup_created_groups();

std::string root_path();
std::string group_prefix();

}  // namespace resctrl

#endif
