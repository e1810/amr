#include "mba_policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

double valid_score(const mba_policy::History &history, std::size_t slot) {
    if (slot >= history.elapsed_ms.size()) {
        return 0.0;
    }
    const double elapsed_ms = history.elapsed_ms[slot];
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0) {
        return 0.0;
    }

    double mb_fraction = 1.0;
    if (slot < history.target_mb_percent.size()) {
        const double percent = history.target_mb_percent[slot];
        if (std::isfinite(percent) && percent > 0.0) {
            mb_fraction = std::clamp(percent / 100.0, 0.01, 1.0);
        }
    }

    return elapsed_ms * mb_fraction;
}

}  // namespace

namespace mba_policy {

Config strict_config() {
    return Config();
}

std::vector<int> plan_levels(const History &history,
                             std::size_t slots,
                             int level_count,
                             const Config &config) {
    (void)config;

    const int max_level = std::max(0, level_count - 1);
    std::vector<int> levels(slots, max_level);
    if (history.elapsed_ms.empty() || level_count <= 1) {
        return levels;
    }

    const std::size_t history_slots = std::min(slots, history.elapsed_ms.size());
    std::size_t critical_slot = history_slots;
    double critical_score = 0.0;
    for (std::size_t slot = 0; slot < history_slots; ++slot) {
        const double score = valid_score(history, slot);
        if (score > critical_score) {
            critical_score = score;
            critical_slot = slot;
        }
    }

    if (critical_slot >= history_slots || critical_score <= 0.0) {
        return levels;
    }

    for (std::size_t slot = 0; slot < history_slots; ++slot) {
        const double score = valid_score(history, slot);
        if (score <= 0.0) {
            levels[slot] = max_level;
            continue;
        }

        const double desired_percent = std::clamp((.9*score + .1*critical_score) / critical_score, 0.0, 1.0) * 100.0;
        static constexpr int kLevelPercents[] = {20, 30, 40, 50, 60, 80, 100};
        int best_level = max_level;
        double best_distance = std::numeric_limits<double>::infinity();
        for (int level = 0; level < level_count; ++level) {
            const int mapped_percent = kLevelPercents[static_cast<std::size_t>(std::clamp(level, 0, static_cast<int>(std::size(kLevelPercents)) - 1))];
            const double distance = std::abs(static_cast<double>(mapped_percent) - desired_percent);
            if (distance < best_distance) {
                best_distance = distance;
                best_level = level;
            }
        }
        levels[slot] = best_level;
    }

    return levels;
}

}  // namespace mba_policy
