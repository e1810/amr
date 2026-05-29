#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

struct Cell {
    int i = 0;
    int j = 0;
    int level = 0;
    double value = 0.0;
    double importance = 0.0;
};

int row_begin(int coarse_n, int parts, int part) {
    return coarse_n * part / parts;
}

int row_end(int coarse_n, int parts, int part) {
    return coarse_n * (part + 1) / parts;
}

int cell_width(int level, int max_level) {
    return 1 << (max_level - level);
}

int owner_part_from_fine_row(int fine_row, int width, int coarse_n, int parts, int fine_n) {
    const int center_fine_row = fine_row + width / 2;
    int coarse_row = center_fine_row * coarse_n / fine_n;
    coarse_row = std::max(0, std::min(coarse_row, coarse_n - 1));

    for (int part = 0; part < parts; ++part) {
        if (row_begin(coarse_n, parts, part) <= coarse_row &&
            coarse_row < row_end(coarse_n, parts, part)) {
            return part;
        }
    }
    return parts - 1;
}

std::string default_output_path(const std::string &input) {
    std::filesystem::path path(input);
    path.replace_extension(".svg");
    return path.string();
}

void prepare_parent_directory(const std::string &path) {
    const std::filesystem::path output_path(path);
    if (!output_path.has_parent_path()) {
        return;
    }

    std::error_code error;
    std::filesystem::create_directories(output_path.parent_path(), error);
    if (error) {
        std::cerr << "failed to create directory: "
                  << output_path.parent_path().string() << '\n';
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0] << " snapshot.txt [output.svg]\n";
        return 2;
    }

    const std::string input = argv[1];
    const std::string output = argc > 2 ? argv[2] : default_output_path(input);

    std::ifstream in(input);
    if (!in) {
        std::cerr << "failed to open input: " << input << '\n';
        return 1;
    }

    int step = 0;
    double time = 0.0;
    int coarse_n = -1;
    int parts = 1;
    int max_level = -1;
    int fine_n = -1;
    std::vector<Cell> cells;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        if (line[0] == '#') {
            std::istringstream header(line.substr(1));
            std::string key;
            header >> key;
            if (key == "step") {
                header >> step;
            } else if (key == "time") {
                header >> time;
            } else if (key == "coarse_n") {
                header >> coarse_n;
            } else if (key == "parts") {
                header >> parts;
            } else if (key == "max_level") {
                header >> max_level;
            } else if (key == "fine_n") {
                header >> fine_n;
            }
            continue;
        }

        std::istringstream row(line);
        Cell cell;
        if (row >> cell.i >> cell.j >> cell.level >> cell.value >> cell.importance) {
            cells.push_back(cell);
        }
    }

    if (coarse_n < 1 || parts < 1 || max_level < 0 || fine_n < 1) {
        std::cerr << "snapshot metadata is incomplete: " << input << '\n';
        return 1;
    }

    prepare_parent_directory(output);
    std::ofstream out(output);
    if (!out) {
        std::cerr << "failed to open output: " << output << '\n';
        return 1;
    }

    constexpr double canvas = 1000.0;
    constexpr double margin = 24.0;
    constexpr double scale = canvas - 2.0 * margin;

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1000 1000\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<title>step " << step << ", time " << std::setprecision(8) << time << "</title>\n";
    out << std::fixed << std::setprecision(3);
    out << "<g stroke=\"#1f2937\" stroke-width=\"0.55\" vector-effect=\"non-scaling-stroke\">\n";

    for (const Cell &cell : cells) {
        const int w = cell_width(cell.level, max_level);
        const double x = margin + static_cast<double>(cell.i) / static_cast<double>(fine_n) * scale;
        const double y = margin + (1.0 - static_cast<double>(cell.j + w) / static_cast<double>(fine_n)) * scale;
        const double size = static_cast<double>(w) / static_cast<double>(fine_n) * scale;
        const int part = owner_part_from_fine_row(cell.j, w, coarse_n, parts, fine_n);
        const double t = std::max(0.0, std::min(1.0, cell.value));
        const int red = static_cast<int>(std::round(239.0 + (220.0 - 239.0) * t));
        const int green = static_cast<int>(std::round(246.0 + (38.0 - 246.0) * t));
        const int blue = static_cast<int>(std::round(255.0 + (38.0 - 255.0) * t));

        out << "<rect x=\"" << x
            << "\" y=\"" << y
            << "\" width=\"" << size
            << "\" height=\"" << size
            << "\" fill=\"rgb(" << red << "," << green << "," << blue << ")"
            << "\"><title>step " << step
            << ", fine(" << cell.i << ", " << cell.j << ")"
            << ", row_part " << part
            << ", level " << cell.level
            << ", value " << cell.value
            << ", importance " << cell.importance
            << "</title></rect>\n";
    }
    out << "</g>\n";

    out << "<g stroke=\"#0f172a\" stroke-width=\"2.0\" vector-effect=\"non-scaling-stroke\">\n";
    out << "<rect x=\"" << margin << "\" y=\"" << margin
        << "\" width=\"" << scale << "\" height=\"" << scale
        << "\" fill=\"none\"/>\n";
    for (int part = 1; part < parts; ++part) {
        const double y_data = static_cast<double>(row_begin(coarse_n, parts, part)) /
                              static_cast<double>(coarse_n);
        const double y = margin + (1.0 - y_data) * scale;
        out << "<line x1=\"" << margin
            << "\" y1=\"" << y
            << "\" x2=\"" << margin + scale
            << "\" y2=\"" << y
            << "\"><title>row-part boundary " << part << "</title></line>\n";
    }
    out << "</g>\n";
    out << "</svg>\n";

    std::cout << "wrote " << output << '\n';
    return 0;
}
