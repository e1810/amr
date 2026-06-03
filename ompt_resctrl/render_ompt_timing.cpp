#include <algorithm>
#include <cmath>
#include <cstdlib>
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
    double target_mhz = 0.0;
    bool resource_applied = false;
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

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, ',')) {
        fields.push_back(field);
    }
    return fields;
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
    return label == "heat_update" || label == "importance";
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
    std::getline(in, line);  // header

    std::map<EventKey, std::vector<TimingRow>> events;
    int max_thread = 0;
    double max_elapsed = 0.0;
    double max_target = 0.0;

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto f = split_csv_line(line);
        if (f.size() < 10) {
            continue;
        }

        TimingRow row;
        row.parallel_id = std::strtoull(f[0].c_str(), nullptr, 10);
        row.region_id = std::atoi(f[1].c_str());
        row.region_exec = std::strtoull(f[2].c_str(), nullptr, 10);
        row.step = static_cast<int>(row.region_exec);
        row.thread_id = std::atoi(f[3].c_str());
        row.team_size = std::atoi(f[4].c_str());
        row.elapsed_ms = std::atof(f[8].c_str());
        row.region_label = amr_region_label(row.region_id);
        if (f.size() > 10) {
            row.cpu_id = std::atoi(f[10].c_str());
        }
        if (f.size() > 11) {
            row.target_mhz = std::atof(f[11].c_str());
        }
        if (f.size() > 12) {
            row.resource_applied = std::atoi(f[12].c_str()) != 0;
        }

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
        max_target = std::max(max_target, row.target_mhz);
    }

    if (events.empty()) {
        std::cerr << "no timing rows selected\n";
        return 1;
    }

    constexpr double left = 190.0;
    constexpr double top = 42.0;
    constexpr double cell_w = 74.0;
    constexpr double row_h = 30.0;
    constexpr double right_pad = 32.0;
    constexpr double bottom_pad = 32.0;
    const double width = left + cell_w * static_cast<double>(max_thread + 1) + right_pad;
    const double height = top + row_h * static_cast<double>(events.size()) + bottom_pad;

    std::ofstream out(output);
    if (!out) {
        std::cerr << "failed to open output: " << output << '\n';
        return 1;
    }

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
        << width << ' ' << height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<style>text{font-family:monospace;font-size:11px}.small{font-size:9px}</style>\n";
    out << "<text x=\"12\" y=\"20\">OMPT resctrl timing, max="
        << std::fixed << std::setprecision(3) << max_elapsed
        << " ms, max_target=" << std::setprecision(0) << max_target << " MHz</text>\n";

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

        out << "<text x=\"8\" y=\"" << y + 18.0 << "\">"
            << "s" << key.step << ' ' << key.region_label << "</text>\n";

        for (const TimingRow &row : rows) {
            const double x = left + cell_w * static_cast<double>(row.thread_id);
            const double ratio = max_elapsed > 0.0 ? row.elapsed_ms / max_elapsed : 0.0;
            const int red = static_cast<int>(std::round(239.0 + (220.0 - 239.0) * ratio));
            const int green = static_cast<int>(std::round(246.0 + (38.0 - 246.0) * ratio));
            const int blue = static_cast<int>(std::round(255.0 + (38.0 - 255.0) * ratio));

            out << "<rect x=\"" << x
                << "\" y=\"" << y
                << "\" width=\"" << cell_w - 3.0
                << "\" height=\"" << row_h - 3.0
                << "\" fill=\"rgb(" << red << ',' << green << ',' << blue << ")\""
                << " stroke=\"#334155\" stroke-width=\"0.35\">"
                << "<title>step " << row.step
                << ", " << row.region_label
                << ", thread " << row.thread_id
                << ", cpu " << row.cpu_id
                << ", target " << std::fixed << std::setprecision(0) << row.target_mhz << " MHz"
                << ", elapsed " << std::setprecision(6) << row.elapsed_ms << " ms"
                << ", applied " << (row.resource_applied ? "yes" : "no")
                << "</title></rect>\n";

            out << "<text class=\"small\" x=\"" << x + 0.5 * (cell_w - 3.0)
                << "\" y=\"" << y + 11.0
                << "\" text-anchor=\"middle\">"
                << std::fixed << std::setprecision(2) << row.elapsed_ms << " ms</text>\n";

            out << "<text class=\"small\" x=\"" << x + 0.5 * (cell_w - 3.0)
                << "\" y=\"" << y + 23.0
                << "\" text-anchor=\"middle\">c" << row.cpu_id
                << ' ' << std::fixed << std::setprecision(0) << row.target_mhz << "M</text>\n";
        }
        ++row_index;
    }

    out << "</svg>\n";
    std::cout << "wrote " << output << '\n';
    return 0;
}
