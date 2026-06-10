#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct TimingRow {
    unsigned long long parallel_id = 0;
    int region_id = 0;
    std::string region_label;
    unsigned long long region_exec = 0;
    int step = 0;
    int thread_id = 0;
    int team_size = 0;
    double elapsed_ms = 0.0;
    int cpu_id = -1;
    double target_value = 0.0;
    std::string resource_kind;
    int target_l3_ways = 0;
    std::string target_l3_mask;
    bool resource_applied = false;
    bool mon_valid = false;
    bool pmu_valid = false;
    long long pmu_llc_references = -1;
    long long pmu_llc_misses = -1;
    long long pmu_llc_hits = -1;
    long long mon_start_llc_occupancy_bytes = -1;
    long long mon_end_llc_occupancy_bytes = -1;
    long long mon_llc_occupancy_delta_bytes = -1;
    long long mon_mbm_total_delta_bytes = -1;
    long long mon_mbm_local_delta_bytes = -1;
    long long mon_mbm_remote_delta_bytes = -1;
};

struct EventKey {
    unsigned long long parallel_id = 0;
    int region_id = 0;
    std::string region_label;
    int step = 0;

    bool operator<(const EventKey &other) const {
        return std::tie(step, parallel_id, region_id, region_label) <
               std::tie(other.step, other.parallel_id, other.region_id, other.region_label);
    }
};

struct Rgb {
    int red = 0;
    int green = 0;
    int blue = 0;
};

Rgb lerp_color(const Rgb &a, const Rgb &b, double u) {
    return {
        static_cast<int>(std::round(static_cast<double>(a.red) +
                                    (static_cast<double>(b.red) - static_cast<double>(a.red)) * u)),
        static_cast<int>(std::round(static_cast<double>(a.green) +
                                    (static_cast<double>(b.green) - static_cast<double>(a.green)) * u)),
        static_cast<int>(std::round(static_cast<double>(a.blue) +
                                    (static_cast<double>(b.blue) - static_cast<double>(a.blue)) * u)),
    };
}

Rgb temperature_color(double value) {
    const double t = std::max(0.0, std::min(1.0, value));
    constexpr int stops = 7;
    const double position[stops] = {0.00, 0.16, 0.32, 0.50, 0.68, 0.84, 1.00};
    const Rgb color[stops] = {
        {8, 22, 120},
        {37, 99, 235},
        {6, 182, 212},
        {34, 197, 94},
        {250, 204, 21},
        {249, 115, 22},
        {220, 38, 38},
    };

    for (int k = 0; k < stops - 1; ++k) {
        if (t <= position[k + 1]) {
            const double u = (t - position[k]) / (position[k + 1] - position[k]);
            return lerp_color(color[k], color[k + 1], u);
        }
    }
    return color[stops - 1];
}

const char *text_color_for_fill(const Rgb &color) {
    const double luminance = 0.2126 * static_cast<double>(color.red) +
                             0.7152 * static_cast<double>(color.green) +
                             0.0722 * static_cast<double>(color.blue);
    return luminance < 120.0 ? "#f8fafc" : "#0f172a";
}

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::map<std::string, std::size_t> header_index(const std::vector<std::string> &header) {
    std::map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < header.size(); ++i) {
        index[header[i]] = i;
    }
    return index;
}

std::string field_or(const std::vector<std::string> &fields,
                     const std::map<std::string, std::size_t> &index,
                     const std::string &name,
                     std::size_t fallback) {
    const auto iter = index.find(name);
    const std::size_t pos = iter != index.end() ? iter->second : fallback;
    return pos < fields.size() ? fields[pos] : "";
}

int int_field(const std::vector<std::string> &fields,
              const std::map<std::string, std::size_t> &index,
              const std::string &name,
              std::size_t fallback) {
    const std::string text = field_or(fields, index, name, fallback);
    return text.empty() ? 0 : std::atoi(text.c_str());
}

double double_field(const std::vector<std::string> &fields,
                    const std::map<std::string, std::size_t> &index,
                    const std::string &name,
                    std::size_t fallback) {
    const std::string text = field_or(fields, index, name, fallback);
    return text.empty() ? 0.0 : std::atof(text.c_str());
}

unsigned long long ull_field(const std::vector<std::string> &fields,
                             const std::map<std::string, std::size_t> &index,
                             const std::string &name,
                             std::size_t fallback) {
    const std::string text = field_or(fields, index, name, fallback);
    return text.empty() ? 0ULL : std::strtoull(text.c_str(), nullptr, 10);
}

long long ll_field(const std::vector<std::string> &fields,
                   const std::map<std::string, std::size_t> &index,
                   const std::string &name,
                   std::size_t fallback) {
    const std::string text = field_or(fields, index, name, fallback);
    return text.empty() ? 0LL : std::strtoll(text.c_str(), nullptr, 10);
}

std::string amr_region_label(int region_id) {
    switch (region_id) {
        case 1: return "init_field";
        case 2: return "heat_update";
        case 3: return "importance";
        case 4: return "mark_refine";
        case 5: return "mark_coarsen";
        default: return "region_" + std::to_string(region_id);
    }
}

bool should_render_region(const std::string &label) {
    return label == "heat_update";
}

std::string target_text(double value, const std::string &resource_kind, bool percent_mode) {
    if (resource_kind == "cat") {
        if (value <= 0.0) {
            return "L3?";
        }
        std::ostringstream out;
        out << "L3 " << std::fixed << std::setprecision(0) << value << "w";
        return out.str();
    }

    if (value <= 0.0) {
        return "?";
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(0) << value;
    if (percent_mode) {
        out << '%';
    } else {
        out << "M";
    }
    return out.str();
}

std::string fixed_text(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

double bytes_to_mib(long long bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double memory_bandwidth_mib_s(const TimingRow &row) {
    if (!row.mon_valid || row.elapsed_ms <= 0.0 || row.mon_mbm_total_delta_bytes < 0) {
        return -1.0;
    }
    return bytes_to_mib(row.mon_mbm_total_delta_bytes) * 1000.0 / row.elapsed_ms;
}

double llc_occupancy_mib(const TimingRow &row) {
    if (!row.mon_valid || row.mon_end_llc_occupancy_bytes < 0) {
        return -1.0;
    }
    return bytes_to_mib(row.mon_end_llc_occupancy_bytes);
}

std::string bandwidth_label(double value) {
    if (value < 0.0) {
        return "n/a";
    }
    if (value >= 1000.0) {
        return fixed_text(value / 1024.0, 2) + " GiB/s";
    }
    return fixed_text(value, 0) + " MiB/s";
}

std::string occupancy_label(double value) {
    if (value < 0.0) {
        return "n/a";
    }
    if (value >= 1.0) {
        return fixed_text(value, 2) + " MiB";
    }
    return fixed_text(value * 1024.0, 0) + " KiB";
}

std::string count_label(double value) {
    if (value < 0.0) {
        return "n/a";
    }
    if (value >= 1000000000.0) {
        return fixed_text(value / 1000000000.0, 2) + "B";
    }
    if (value >= 1000000.0) {
        return fixed_text(value / 1000000.0, 1) + "M";
    }
    if (value >= 1000.0) {
        return fixed_text(value / 1000.0, 1) + "K";
    }
    return fixed_text(value, 0);
}

std::string derived_output_path(const std::string &output, const std::string &suffix) {
    const std::string timing_svg = "timing.svg";
    const std::size_t timing_pos = output.rfind(timing_svg);
    if (timing_pos != std::string::npos && timing_pos + timing_svg.size() == output.size()) {
        return output.substr(0, timing_pos) + suffix + ".svg";
    }

    const std::size_t dot_pos = output.rfind(".svg");
    if (dot_pos != std::string::npos && dot_pos + 4 == output.size()) {
        return output.substr(0, dot_pos) + "_" + suffix + ".svg";
    }
    return output + "_" + suffix + ".svg";
}

template <typename ValueFunc, typename LabelFunc, typename DetailFunc>
bool render_metric_svg(const std::map<EventKey, std::vector<TimingRow>> &events,
                       int max_thread,
                       const std::string &output,
                       const std::string &title,
                       const std::string &units,
                       bool percent_mode,
                       ValueFunc value_func,
                       LabelFunc label_func,
                       DetailFunc detail_func) {
    double max_value = 0.0;
    bool has_value = false;
    for (const auto &entry : events) {
        for (const TimingRow &row : entry.second) {
            const double value = value_func(row);
            if (value >= 0.0) {
                has_value = true;
                max_value = std::max(max_value, value);
            }
        }
    }

    constexpr double left = 76.0;
    constexpr double top = 42.0;
    constexpr double cell_w = 112.0;
    constexpr double row_h = 30.0;
    constexpr double right_pad = 32.0;
    constexpr double bottom_pad = 32.0;
    const double width = left + cell_w * static_cast<double>(max_thread + 1) + right_pad;
    const double height = top + row_h * static_cast<double>(events.size()) + bottom_pad;

    std::ofstream out(output);
    if (!out) {
        std::cerr << "failed to open output: " << output
                  << ": " << std::strerror(errno) << '\n';
        return false;
    }

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
        << width << ' ' << height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<style>text{font-family:monospace;font-size:11px}.small{font-size:9px}</style>\n";
    out << "<text x=\"12\" y=\"20\">" << title
        << ", max=" << (has_value ? label_func(max_value) : std::string("n/a"))
        << ", color=0-" << (has_value ? label_func(max_value) : std::string("n/a"))
        << "</text>\n";

    for (int tid = 0; tid <= max_thread; ++tid) {
        const double x = left + cell_w * static_cast<double>(tid) + cell_w * 0.5;
        out << "<text class=\"small\" x=\"" << x
            << "\" y=\"34\" text-anchor=\"middle\">t" << tid << "</text>\n";
    }

    int row_index = 0;
    for (const auto &entry : events) {
        const EventKey &key = entry.first;
        const auto &rows = entry.second;
        const double y = top + row_h * static_cast<double>(row_index);
        double event_max = -1.0;
        for (const TimingRow &row : rows) {
            event_max = std::max(event_max, value_func(row));
        }

        out << "<text x=\"8\" y=\"" << y + 11.0 << "\">"
            << "s" << key.step << "</text>\n";
        out << "<text class=\"small\" x=\"8\" y=\"" << y + 24.0 << "\">"
            << (event_max >= 0.0 ? label_func(event_max) : std::string("n/a"))
            << "</text>\n";

        for (const TimingRow &row : rows) {
            const double x = left + cell_w * static_cast<double>(row.thread_id);
            const double value = value_func(row);
            const bool valid = value >= 0.0;
            const double ratio = valid && max_value > 0.0 ? value / max_value : 0.0;
            const Rgb color = valid ? temperature_color(ratio) : Rgb{226, 232, 240};
            const char *label_color = valid ? text_color_for_fill(color) : "#475569";

            out << "<rect x=\"" << x
                << "\" y=\"" << y
                << "\" width=\"" << cell_w - 3.0
                << "\" height=\"" << row_h - 3.0
                << "\" fill=\"rgb(" << color.red << ',' << color.green << ',' << color.blue << ")\""
                << " stroke=\"#334155\" stroke-width=\"0.35\">"
                << "<title>step " << row.step
                << ", " << row.region_label
                << ", thread " << row.thread_id
                << ", cpu " << row.cpu_id
                << ", " << units << ' ' << (valid ? fixed_text(value, 6) : std::string("n/a"))
                << ", target " << target_text(row.target_value, row.resource_kind, percent_mode)
                << (row.resource_kind == "cat" ? ", mask " + row.target_l3_mask : "")
                << detail_func(row)
                << "</title></rect>\n";

            out << "<text class=\"small\" x=\"" << x + 0.5 * (cell_w - 3.0)
                << "\" y=\"" << y + 11.0
                << "\" text-anchor=\"middle\" fill=\"" << label_color << "\">"
                << label_func(value) << "</text>\n";

            out << "<text class=\"small\" x=\"" << x + 0.5 * (cell_w - 3.0)
                << "\" y=\"" << y + 23.0
                << "\" text-anchor=\"middle\" fill=\"" << label_color << "\">"
                << target_text(row.target_value, row.resource_kind, percent_mode)
                << "</text>\n";
        }
        ++row_index;
    }

    out << "</svg>\n";
    std::cout << "wrote " << output << '\n';
    return true;
}


bool render_cache_counts_svg(const std::map<EventKey, std::vector<TimingRow>> &events,
                             int max_thread,
                             const std::string &output,
                             bool percent_mode) {
    double max_hit = 0.0;
    double max_miss = 0.0;
    bool has_value = false;
    for (const auto &entry : events) {
        for (const TimingRow &row : entry.second) {
            if (row.pmu_valid && row.pmu_llc_hits >= 0) {
                has_value = true;
                max_hit = std::max(max_hit, static_cast<double>(row.pmu_llc_hits));
            }
            if (row.pmu_valid && row.pmu_llc_misses >= 0) {
                has_value = true;
                max_miss = std::max(max_miss, static_cast<double>(row.pmu_llc_misses));
            }
        }
    }

    const double max_value = std::max(max_hit, max_miss);
    constexpr double left = 86.0;
    constexpr double top = 46.0;
    constexpr double cell_w = 128.0;
    constexpr double row_h = 48.0;
    constexpr double right_pad = 32.0;
    constexpr double bottom_pad = 32.0;
    const double width = left + cell_w * static_cast<double>(max_thread + 1) + right_pad;
    const double height = top + row_h * static_cast<double>(events.size()) + bottom_pad;

    std::ofstream out(output);
    if (!out) {
        std::cerr << "failed to open output: " << output
                  << ": " << std::strerror(errno) << '\n';
        return false;
    }

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
        << width << ' ' << height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<style>text{font-family:monospace;font-size:11px}.small{font-size:9px}</style>\n";
    out << "<text x=\"12\" y=\"20\">OMPT resctrl LLC cache counts, max_hit="
        << (has_value ? count_label(max_hit) : std::string("n/a"))
        << ", max_miss=" << (has_value ? count_label(max_miss) : std::string("n/a"))
        << ", color=0-" << (has_value ? count_label(max_value) : std::string("n/a"))
        << "</text>\n";
    out << "<text class=\"small\" x=\"12\" y=\"35\">top: hit count, bottom: miss count</text>\n";

    for (int tid = 0; tid <= max_thread; ++tid) {
        const double x = left + cell_w * static_cast<double>(tid) + cell_w * 0.5;
        out << "<text class=\"small\" x=\"" << x
            << "\" y=\"40\" text-anchor=\"middle\">t" << tid << "</text>\n";
    }

    int row_index = 0;
    for (const auto &entry : events) {
        const EventKey &key = entry.first;
        const auto &rows = entry.second;
        const double y = top + row_h * static_cast<double>(row_index);
        double event_max_hit = -1.0;
        double event_max_miss = -1.0;
        for (const TimingRow &row : rows) {
            if (row.pmu_valid && row.pmu_llc_hits >= 0) {
                event_max_hit = std::max(event_max_hit, static_cast<double>(row.pmu_llc_hits));
            }
            if (row.pmu_valid && row.pmu_llc_misses >= 0) {
                event_max_miss = std::max(event_max_miss, static_cast<double>(row.pmu_llc_misses));
            }
        }

        out << "<text x=\"8\" y=\"" << y + 12.0 << "\">"
            << "s" << key.step << "</text>\n";
        out << "<text class=\"small\" x=\"8\" y=\"" << y + 27.0 << "\">H "
            << count_label(event_max_hit) << "</text>\n";
        out << "<text class=\"small\" x=\"8\" y=\"" << y + 40.0 << "\">M "
            << count_label(event_max_miss) << "</text>\n";

        for (const TimingRow &row : rows) {
            const double x = left + cell_w * static_cast<double>(row.thread_id);
            const double half_h = (row_h - 4.0) * 0.5;
            const bool hit_valid = row.pmu_valid && row.pmu_llc_hits >= 0;
            const bool miss_valid = row.pmu_valid && row.pmu_llc_misses >= 0;
            const double hit_value = hit_valid ? static_cast<double>(row.pmu_llc_hits) : -1.0;
            const double miss_value = miss_valid ? static_cast<double>(row.pmu_llc_misses) : -1.0;
            const double hit_ratio = hit_valid && max_value > 0.0 ? hit_value / max_value : 0.0;
            const double miss_ratio = miss_valid && max_value > 0.0 ? miss_value / max_value : 0.0;
            const Rgb hit_color = hit_valid ? temperature_color(hit_ratio) : Rgb{226, 232, 240};
            const Rgb miss_color = miss_valid ? temperature_color(miss_ratio) : Rgb{226, 232, 240};
            const char *hit_label_color = hit_valid ? text_color_for_fill(hit_color) : "#475569";
            const char *miss_label_color = miss_valid ? text_color_for_fill(miss_color) : "#475569";
            const double miss_rate = row.pmu_valid && row.pmu_llc_references > 0
                                         ? 100.0 * static_cast<double>(row.pmu_llc_misses) /
                                               static_cast<double>(row.pmu_llc_references)
                                         : -1.0;

            const std::string tooltip_prefix =
                "step " + std::to_string(row.step) +
                ", " + row.region_label +
                ", thread " + std::to_string(row.thread_id) +
                ", cpu " + std::to_string(row.cpu_id) +
                ", references " + std::to_string(row.pmu_llc_references) +
                ", hits " + std::to_string(row.pmu_llc_hits) +
                ", misses " + std::to_string(row.pmu_llc_misses) +
                ", miss_rate_pct " + (miss_rate >= 0.0 ? fixed_text(miss_rate, 3) : std::string("n/a")) +
                ", target " + target_text(row.target_value, row.resource_kind, percent_mode) +
                (row.resource_kind == "cat" ? ", mask " + row.target_l3_mask : "");

            out << "<rect x=\"" << x
                << "\" y=\"" << y
                << "\" width=\"" << cell_w - 3.0
                << "\" height=\"" << half_h
                << "\" fill=\"rgb(" << hit_color.red << ',' << hit_color.green << ',' << hit_color.blue << ")\""
                << " stroke=\"#334155\" stroke-width=\"0.35\">"
                << "<title>" << tooltip_prefix << ", metric hits</title></rect>\n";
            out << "<rect x=\"" << x
                << "\" y=\"" << y + half_h
                << "\" width=\"" << cell_w - 3.0
                << "\" height=\"" << half_h
                << "\" fill=\"rgb(" << miss_color.red << ',' << miss_color.green << ',' << miss_color.blue << ")\""
                << " stroke=\"#334155\" stroke-width=\"0.35\">"
                << "<title>" << tooltip_prefix << ", metric misses</title></rect>\n";

            out << "<text class=\"small\" x=\"" << x + 0.5 * (cell_w - 3.0)
                << "\" y=\"" << y + 13.0
                << "\" text-anchor=\"middle\" fill=\"" << hit_label_color << "\">H "
                << count_label(hit_value) << "</text>\n";
            out << "<text class=\"small\" x=\"" << x + 0.5 * (cell_w - 3.0)
                << "\" y=\"" << y + half_h + 13.0
                << "\" text-anchor=\"middle\" fill=\"" << miss_label_color << "\">M "
                << count_label(miss_value) << "</text>\n";
        }
        ++row_index;
    }

    out << "</svg>\n";
    std::cout << "wrote " << output << '\n';
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: " << argv[0] << " ompt_regions.csv output.svg [exec_interval]\n";
        return 2;
    }

    const std::string input = argv[1];
    const std::string output = argv[2];
    const int step_interval = argc > 3 ? std::max(1, std::atoi(argv[3])) : 1;

    std::ifstream in(input);
    if (!in) {
        std::cerr << "failed to open input: " << input << '\n';
        return 1;
    }

    std::string line;
    if (!std::getline(in, line)) {
        std::cerr << "empty input: " << input << '\n';
        return 1;
    }

    const auto header = split_csv_line(line);
    const auto columns = header_index(header);
    const bool percent_mode = columns.find("target_mb_percent") != columns.end();
    const bool has_resource_kind = columns.find("resource_kind") != columns.end();
    const bool has_l3_target = columns.find("target_l3_ways") != columns.end();
    const bool has_monitor = columns.find("mon_valid") != columns.end();
    const bool has_pmu = columns.find("pmu_valid") != columns.end();
    const std::string target_column = percent_mode ? "target_mb_percent" : "target_mhz";

    std::map<EventKey, std::vector<TimingRow>> events;
    int max_thread = 0;
    double max_elapsed = 0.0;
    double max_target = 0.0;
    bool saw_cat = false;

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto f = split_csv_line(line);
        if (f.size() < 10) {
            continue;
        }

        TimingRow row;
        row.parallel_id = ull_field(f, columns, "parallel_id", 0);
        row.region_id = int_field(f, columns, "region_id", 1);
        row.region_exec = ull_field(f, columns, "region_exec", 2);
        row.step = static_cast<int>(row.region_exec);
        row.thread_id = int_field(f, columns, "thread_id", 3);
        row.team_size = int_field(f, columns, "team_size", 4);
        row.elapsed_ms = double_field(f, columns, "elapsed_ms", 8);
        row.region_label = amr_region_label(row.region_id);
        row.cpu_id = int_field(f, columns, "cpu_id", 10);
        row.resource_kind = has_resource_kind ? field_or(f, columns, "resource_kind", 17) : "mba";
        if (row.resource_kind == "cat") {
            saw_cat = true;
        }
        row.target_l3_ways = has_l3_target ? int_field(f, columns, "target_l3_ways", 18) : 0;
        row.target_l3_mask = has_l3_target ? field_or(f, columns, "target_l3_mask", 19) : "";
        row.target_value = row.resource_kind == "cat"
                               ? static_cast<double>(row.target_l3_ways)
                               : double_field(f, columns, target_column, 11);
        row.resource_applied = int_field(f, columns, "resource_applied", percent_mode ? 12 : 13) != 0;
        row.pmu_valid = has_pmu && int_field(f, columns, "pmu_valid", 10) != 0;
        row.pmu_llc_references = has_pmu ? ll_field(f, columns, "pmu_llc_references", 11) : -1;
        row.pmu_llc_misses = has_pmu ? ll_field(f, columns, "pmu_llc_misses", 12) : -1;
        row.pmu_llc_hits = has_pmu ? ll_field(f, columns, "pmu_llc_hits", 13) : -1;
        row.mon_valid = has_monitor && int_field(f, columns, "mon_valid", 20) != 0;
        row.mon_start_llc_occupancy_bytes = has_monitor ? ll_field(f, columns, "mon_start_llc_occupancy_bytes", 22) : -1;
        row.mon_end_llc_occupancy_bytes = has_monitor ? ll_field(f, columns, "mon_end_llc_occupancy_bytes", 23) : -1;
        row.mon_llc_occupancy_delta_bytes = has_monitor ? ll_field(f, columns, "mon_llc_occupancy_delta_bytes", 24) : -1;
        row.mon_mbm_total_delta_bytes = has_monitor ? ll_field(f, columns, "mon_mbm_total_delta_bytes", 27) : -1;
        row.mon_mbm_local_delta_bytes = has_monitor ? ll_field(f, columns, "mon_mbm_local_delta_bytes", 30) : -1;
        row.mon_mbm_remote_delta_bytes = has_monitor ? ll_field(f, columns, "mon_mbm_remote_delta_bytes", 33) : -1;

        if (!should_render_region(row.region_label) ||
            (row.step != 0 && row.step % step_interval != 0)) {
            continue;
        }

        EventKey key;
        key.parallel_id = row.parallel_id;
        key.region_id = row.region_id;
        key.region_label = row.region_label;
        key.step = row.step;
        events[key].push_back(row);

        max_thread = std::max(max_thread, row.thread_id);
        max_elapsed = std::max(max_elapsed, row.elapsed_ms);
        max_target = std::max(max_target, row.target_value);
    }

    if (events.empty()) {
        std::cerr << "no timing rows selected\n";
        return 1;
    }

    constexpr double left = 76.0;
    constexpr double top = 42.0;
    constexpr double cell_w = 96.0;
    constexpr double row_h = 30.0;
    constexpr double right_pad = 32.0;
    constexpr double bottom_pad = 32.0;
    constexpr double color_min_ms = 20.0;
    constexpr double color_max_ms = 45.0;
    const double width = left + cell_w * static_cast<double>(max_thread + 1) + right_pad;
    const double height = top + row_h * static_cast<double>(events.size()) + bottom_pad;

    std::ofstream out(output);
    if (!out) {
        std::cerr << "failed to open output: " << output
                  << ": " << std::strerror(errno) << '\n';
        return 1;
    }

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
        << width << ' ' << height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<style>text{font-family:monospace;font-size:11px}.small{font-size:9px}</style>\n";
    out << "<text x=\"12\" y=\"20\">OMPT resctrl timing, max="
        << std::fixed << std::setprecision(3) << max_elapsed
        << " ms, color=20-45 ms, max_target=" << target_text(max_target, saw_cat ? "cat" : "mba", percent_mode)
        << "</text>\n";

    for (int tid = 0; tid <= max_thread; ++tid) {
        const double x = left + cell_w * static_cast<double>(tid) + cell_w * 0.5;
        out << "<text class=\"small\" x=\"" << x
            << "\" y=\"34\" text-anchor=\"middle\">t" << tid << "</text>\n";
    }

    int row_index = 0;
    for (const auto &entry : events) {
        const EventKey &key = entry.first;
        const auto &rows = entry.second;
        const double y = top + row_h * static_cast<double>(row_index);
        double event_max_ms = 0.0;
        for (const TimingRow &row : rows) {
            event_max_ms = std::max(event_max_ms, row.elapsed_ms);
        }

        out << "<text x=\"8\" y=\"" << y + 11.0 << "\">"
            << "s" << key.step << "</text>\n";
        out << "<text class=\"small\" x=\"8\" y=\"" << y + 24.0 << "\">"
            << std::fixed << std::setprecision(2) << event_max_ms << " ms</text>\n";

        for (const TimingRow &row : rows) {
            const double x = left + cell_w * static_cast<double>(row.thread_id);
            const double ratio = (row.elapsed_ms - color_min_ms) / (color_max_ms - color_min_ms);
            const Rgb color = temperature_color(ratio);
            const char *label_color = text_color_for_fill(color);

            out << "<rect x=\"" << x
                << "\" y=\"" << y
                << "\" width=\"" << cell_w - 3.0
                << "\" height=\"" << row_h - 3.0
                << "\" fill=\"rgb(" << color.red << ',' << color.green << ',' << color.blue << ")\""
                << " stroke=\"#334155\" stroke-width=\"0.35\">"
                << "<title>step " << row.step
                << ", " << row.region_label
                << ", thread " << row.thread_id
                << ", cpu " << row.cpu_id
                << ", target " << target_text(row.target_value, row.resource_kind, percent_mode)
                << (row.resource_kind == "cat" ? ", mask " + row.target_l3_mask : "")
                << ", elapsed " << std::setprecision(6) << row.elapsed_ms << " ms"
                << ", applied " << (row.resource_applied ? "yes" : "no")
                << "</title></rect>\n";

            out << "<text class=\"small\" x=\"" << x + 0.5 * (cell_w - 3.0)
                << "\" y=\"" << y + 11.0
                << "\" text-anchor=\"middle\" fill=\"" << label_color << "\">"
                << std::fixed << std::setprecision(2) << row.elapsed_ms << " ms</text>\n";

            out << "<text class=\"small\" x=\"" << x + 0.5 * (cell_w - 3.0)
                << "\" y=\"" << y + 23.0
                << "\" text-anchor=\"middle\" fill=\"" << label_color << "\">"
                << target_text(row.target_value, row.resource_kind, percent_mode)
                << (row.resource_applied ? "" : "*")
                << "</text>\n";
        }
        ++row_index;
    }

    out << "</svg>\n";
    std::cout << "wrote " << output << '\n';

    const std::string bandwidth_output = derived_output_path(output, "memory_bandwidth");
    const std::string llc_output = derived_output_path(output, "llc_occupancy");
    const std::string cache_output = derived_output_path(output, "cache_counts");

    const auto bandwidth_detail = [](const TimingRow &row) {
        std::ostringstream detail;
        detail << ", elapsed_ms " << std::fixed << std::setprecision(6) << row.elapsed_ms
               << ", mbm_total_delta_bytes " << row.mon_mbm_total_delta_bytes
               << ", mbm_local_delta_bytes " << row.mon_mbm_local_delta_bytes
               << ", mbm_remote_delta_bytes " << row.mon_mbm_remote_delta_bytes;
        return detail.str();
    };
    const auto llc_detail = [](const TimingRow &row) {
        std::ostringstream detail;
        detail << ", start_llc_occupancy_bytes " << row.mon_start_llc_occupancy_bytes
               << ", end_llc_occupancy_bytes " << row.mon_end_llc_occupancy_bytes
               << ", llc_occupancy_delta_bytes " << row.mon_llc_occupancy_delta_bytes;
        return detail.str();
    };

    if (!render_metric_svg(events, max_thread, bandwidth_output,
                           "OMPT resctrl memory bandwidth", "bandwidth_mib_s", percent_mode,
                           memory_bandwidth_mib_s, bandwidth_label, bandwidth_detail)) {
        return 1;
    }
    if (!render_metric_svg(events, max_thread, llc_output,
                           "OMPT resctrl LLC occupancy", "llc_occupancy_mib", percent_mode,
                           llc_occupancy_mib, occupancy_label, llc_detail)) {
        return 1;
    }
    if (has_pmu && !render_cache_counts_svg(events, max_thread, cache_output, percent_mode)) {
        return 1;
    }
    return 0;
}
