#include "dense_amr_output.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {

bool ends_with(const std::string &text, const std::string &suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

void write_output(const DenseAmr &amr, const std::string &path) {
    const int leaves = count_leaves(amr);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    if (ends_with(path, ".vtk")) {
        out << "# vtk DataFile Version 3.0\n";
        out << "modular dense fine-grid AMR\n";
        out << "ASCII\n";
        out << "DATASET UNSTRUCTURED_GRID\n";
        out << "POINTS " << leaves * 4 << " double\n";
        out << std::setprecision(17);

        for (int j = 0; j < amr.fine_n; ++j) {
            for (int i = 0; i < amr.fine_n; ++i) {
                const int p = index2d(i, j, amr.fine_n);
                if (!amr.active[static_cast<std::size_t>(p)]) {
                    continue;
                }
                const int w = cell_width(amr.level[static_cast<std::size_t>(p)], amr.max_level);
                const double x0 = static_cast<double>(i) / static_cast<double>(amr.fine_n);
                const double y0 = static_cast<double>(j) / static_cast<double>(amr.fine_n);
                const double size = static_cast<double>(w) / static_cast<double>(amr.fine_n);
                out << x0 << ' ' << y0 << " 0\n";
                out << x0 + size << ' ' << y0 << " 0\n";
                out << x0 + size << ' ' << y0 + size << " 0\n";
                out << x0 << ' ' << y0 + size << " 0\n";
            }
        }

        out << "CELLS " << leaves << ' ' << leaves * 5 << '\n';
        for (int c = 0; c < leaves; ++c) {
            const int p = c * 4;
            out << "4 " << p << ' ' << p + 1 << ' ' << p + 2 << ' ' << p + 3 << '\n';
        }
        out << "CELL_TYPES " << leaves << '\n';
        for (int c = 0; c < leaves; ++c) {
            out << "9\n";
        }
        out << "CELL_DATA " << leaves << '\n';
        out << "SCALARS value double 1\n";
        out << "LOOKUP_TABLE default\n";
        for (int j = 0; j < amr.fine_n; ++j) {
            for (int i = 0; i < amr.fine_n; ++i) {
                const int p = index2d(i, j, amr.fine_n);
                if (amr.active[static_cast<std::size_t>(p)]) {
                    out << amr.value[static_cast<std::size_t>(p)] << '\n';
                }
            }
        }
        out << "SCALARS level int 1\n";
        out << "LOOKUP_TABLE default\n";
        for (int j = 0; j < amr.fine_n; ++j) {
            for (int i = 0; i < amr.fine_n; ++i) {
                const int p = index2d(i, j, amr.fine_n);
                if (amr.active[static_cast<std::size_t>(p)]) {
                    out << static_cast<int>(amr.level[static_cast<std::size_t>(p)]) << '\n';
                }
            }
        }
        out << "SCALARS importance double 1\n";
        out << "LOOKUP_TABLE default\n";
        for (int j = 0; j < amr.fine_n; ++j) {
            for (int i = 0; i < amr.fine_n; ++i) {
                const int p = index2d(i, j, amr.fine_n);
                if (amr.active[static_cast<std::size_t>(p)]) {
                    out << amr.importance[static_cast<std::size_t>(p)] << '\n';
                }
            }
        }
        return;
    }

    constexpr double margin = 24.0;
    constexpr double scale = 952.0;

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1000 1000\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << std::fixed << std::setprecision(3);
    out << "<g stroke=\"#1f2937\" stroke-width=\"0.55\" vector-effect=\"non-scaling-stroke\">\n";

    for (int j = 0; j < amr.fine_n; ++j) {
        for (int i = 0; i < amr.fine_n; ++i) {
            const int p = index2d(i, j, amr.fine_n);
            if (!amr.active[static_cast<std::size_t>(p)]) {
                continue;
            }

            const int lev = amr.level[static_cast<std::size_t>(p)];
            const int w = cell_width(lev, amr.max_level);
            const double x = margin + static_cast<double>(i) / static_cast<double>(amr.fine_n) * scale;
            const double y = margin + (1.0 - static_cast<double>(j + w) / static_cast<double>(amr.fine_n)) * scale;
            const double size = static_cast<double>(w) / static_cast<double>(amr.fine_n) * scale;
            const int part = owner_part_from_fine_row(j, w, amr.coarse_n, amr.parts, amr.fine_n);
            const double t = std::max(0.0, std::min(1.0, amr.value[static_cast<std::size_t>(p)]));
            const int red = static_cast<int>(std::round(239.0 + (220.0 - 239.0) * t));
            const int green = static_cast<int>(std::round(246.0 + (38.0 - 246.0) * t));
            const int blue = static_cast<int>(std::round(255.0 + (38.0 - 255.0) * t));

            out << "<rect x=\"" << x
                << "\" y=\"" << y
                << "\" width=\"" << size
                << "\" height=\"" << size
                << "\" fill=\"rgb(" << red << "," << green << "," << blue << ")"
                << "\"><title>fine(" << i << ", " << j << ")"
                << ", row_part " << part
                << ", level " << lev
                << ", value " << amr.value[static_cast<std::size_t>(p)]
                << ", importance " << amr.importance[static_cast<std::size_t>(p)]
                << "</title></rect>\n";
        }
    }
    out << "</g>\n";

    out << "<g stroke=\"#0f172a\" stroke-width=\"2.0\" vector-effect=\"non-scaling-stroke\">\n";
    out << "<rect x=\"" << margin << "\" y=\"" << margin
        << "\" width=\"" << scale << "\" height=\"" << scale
        << "\" fill=\"none\"/>\n";
    for (int part = 1; part < amr.parts; ++part) {
        const double y_data = static_cast<double>(row_begin(amr.coarse_n, amr.parts, part)) /
                              static_cast<double>(amr.coarse_n);
        const double y = margin + (1.0 - y_data) * scale;
        out << "<line x1=\"" << margin
            << "\" y1=\"" << y
            << "\" x2=\"" << margin + scale
            << "\" y2=\"" << y
            << "\"><title>row-part boundary " << part << "</title></line>\n";
    }
    out << "</g>\n";
    out << "</svg>\n";
}
