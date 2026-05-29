#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

int index2d(int i, int j, int n) {
    return j * n + i;
}

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

bool ends_with(const std::string &text, const std::string &suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void rebuild_owner(const std::vector<unsigned char> &active,
                   const std::vector<unsigned char> &level,
                   std::vector<int> &owner,
                   int fine_n,
                   int max_level) {
    std::fill(owner.begin(), owner.end(), -1);

    for (int j = 0; j < fine_n; ++j) {
        for (int i = 0; i < fine_n; ++i) {
            const int p = index2d(i, j, fine_n);
            if (!active[static_cast<std::size_t>(p)]) {
                continue;
            }

            const int w = cell_width(level[static_cast<std::size_t>(p)], max_level);
            for (int jj = j; jj < j + w; ++jj) {
                for (int ii = i; ii < i + w; ++ii) {
                    owner[static_cast<std::size_t>(index2d(ii, jj, fine_n))] = p;
                }
            }
        }
    }
}

double face_average(const std::vector<int> &owner,
                    const std::vector<double> &value,
                    int fine_n,
                    int self,
                    int i,
                    int j,
                    int w,
                    int direction,
                    double fallback) {
    std::vector<int> seen;
    double sum = 0.0;

    for (int s = 0; s < w; ++s) {
        int ni = i;
        int nj = j;

        if (direction == 0) {       // left
            ni = i - 1;
            nj = j + s;
        } else if (direction == 1) {  // right
            ni = i + w;
            nj = j + s;
        } else if (direction == 2) {  // bottom
            ni = i + s;
            nj = j - 1;
        } else {                    // top
            ni = i + s;
            nj = j + w;
        }

        if (ni < 0 || ni >= fine_n || nj < 0 || nj >= fine_n) {
            continue;
        }

        const int q = owner[static_cast<std::size_t>(index2d(ni, nj, fine_n))];
        if (q < 0 || q == self || std::find(seen.begin(), seen.end(), q) != seen.end()) {
            continue;
        }

        seen.push_back(q);
        sum += value[static_cast<std::size_t>(q)];
    }

    return seen.empty() ? fallback : sum / static_cast<double>(seen.size());
}

int main(int argc, char **argv) {
    // ------------------------------------------------------------------
    // This file is the "dense fine-grid" version.
    //
    // coarse_n: the initial coarse grid size, e.g. 16 x 16.
    // fine_n:   the finest grid used as the storage canvas.
    //           fine_n = coarse_n * 2^max_level.
    //
    // Each array below has fine_n x fine_n entries. Only entries with
    // active[p] == 1 are current AMR leaves.
    // ------------------------------------------------------------------
    const int coarse_n = argc > 1 ? std::atoi(argv[1]) : 16;
    const int parts = argc > 2 ? std::atoi(argv[2]) : 4;
    const int max_level = argc > 3 ? std::atoi(argv[3]) : 3;
    const double radius = argc > 4 ? std::atof(argv[4]) : 0.28;
    const std::string output = argc > 5 ? argv[5] : "amr_dense.svg";

    if (coarse_n < 1 || parts < 1 || parts > coarse_n || max_level < 0 || max_level > 10 ||
        radius <= 0.0) {
        std::cerr << "usage: " << argv[0]
                  << " [coarse_n>=1] [parts in 1..coarse_n] [max_level 0..10]"
                  << " [radius>0] [output.svg|output.vtk]\n";
        return 2;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(parts);
    std::cout << "OpenMP max threads: " << omp_get_max_threads()
              << ", requested threads: " << parts << '\n';

    const int refine_factor = 1 << max_level;
    const int fine_n = coarse_n * refine_factor;
    const int coarse_width = refine_factor;
    const int array_size = fine_n * fine_n;

    std::vector<unsigned char> active(static_cast<std::size_t>(array_size), 0);
    std::vector<unsigned char> level(static_cast<std::size_t>(array_size), 0);
    std::vector<unsigned char> refine_mark(static_cast<std::size_t>(array_size), 0);
    std::vector<int> owner(static_cast<std::size_t>(array_size), -1);
    std::vector<double> value(static_cast<std::size_t>(array_size), 0.0);
    std::vector<double> next_value(static_cast<std::size_t>(array_size), 0.0);
    std::vector<double> importance(static_cast<std::size_t>(array_size), 0.0);

    // ------------------------------------------------------------------
    // Initial coarse_n x coarse_n cells. A coarse cell is represented by its
    // lower-left corner on the fine grid.
    // ------------------------------------------------------------------
    for (int coarse_j = 0; coarse_j < coarse_n; ++coarse_j) {
        for (int coarse_i = 0; coarse_i < coarse_n; ++coarse_i) {
            const int i = coarse_i * coarse_width;
            const int j = coarse_j * coarse_width;
            const int p = index2d(i, j, fine_n);
            active[static_cast<std::size_t>(p)] = 1;
            level[static_cast<std::size_t>(p)] = 0;
        }
    }

    std::cout << "initial: coarse_grid=" << coarse_n << "x" << coarse_n
              << ", fine_grid=" << fine_n << "x" << fine_n
              << ", row_parts=" << parts << '\n';
    for (int part = 0; part < parts; ++part) {
        std::cout << "static part " << part << ": coarse rows ["
                  << row_begin(coarse_n, parts, part) << ", "
                  << row_end(coarse_n, parts, part) << ")\n";
    }

    // ------------------------------------------------------------------
    // 1a. Initial field. OpenMP distributes fine-grid rows statically.
    // ------------------------------------------------------------------
#pragma omp parallel for schedule(static)
    for (int j = 0; j < fine_n; ++j) {
        for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (!active[static_cast<std::size_t>(p)]) {
                    continue;
                }

                const int w = cell_width(level[static_cast<std::size_t>(p)], max_level);
                const double x0 = static_cast<double>(i) / static_cast<double>(fine_n);
                const double y0 = static_cast<double>(j) / static_cast<double>(fine_n);
                const double size = static_cast<double>(w) / static_cast<double>(fine_n);
                const double x1 = x0 + size;
                const double y1 = y0 + size;
                const double nearest_x = std::max(x0, std::min(0.5, x1));
                const double nearest_y = std::max(y0, std::min(0.5, y1));
                const double dx = nearest_x - 0.5;
                const double dy = nearest_y - 0.5;
                value[static_cast<std::size_t>(p)] = (dx * dx + dy * dy <= radius * radius) ? 1.0 : 0.0;
        }
    }

    // ------------------------------------------------------------------
    // 1b. Dummy update using actual neighbors.
    // owner[q] tells which active AMR cell owns each finest-grid location.
    // For each active cell, sample the owners touching its four faces.
    // ------------------------------------------------------------------
    rebuild_owner(active, level, owner, fine_n, max_level);

#pragma omp parallel for schedule(static)
    for (int j = 0; j < fine_n; ++j) {
        for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (!active[static_cast<std::size_t>(p)]) {
                    continue;
                }

                const int w = cell_width(level[static_cast<std::size_t>(p)], max_level);
                const double center = value[static_cast<std::size_t>(p)];
                const double neighbor_average =
                    0.25 * (face_average(owner, value, fine_n, p, i, j, w, 0, center) +
                            face_average(owner, value, fine_n, p, i, j, w, 1, center) +
                            face_average(owner, value, fine_n, p, i, j, w, 2, center) +
                            face_average(owner, value, fine_n, p, i, j, w, 3, center));
                next_value[static_cast<std::size_t>(p)] = center + 0.02 * (neighbor_average - center);
        }
    }

#pragma omp parallel for schedule(static)
    for (int j = 0; j < fine_n; ++j) {
        for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (active[static_cast<std::size_t>(p)]) {
                    value[static_cast<std::size_t>(p)] = next_value[static_cast<std::size_t>(p)];
                }
        }
    }

    // ------------------------------------------------------------------
    // 2. Importance. Deliberately simple: importance = abs(value).
    // ------------------------------------------------------------------
#pragma omp parallel for schedule(static)
    for (int j = 0; j < fine_n; ++j) {
        for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (active[static_cast<std::size_t>(p)]) {
                    importance[static_cast<std::size_t>(p)] = std::abs(value[static_cast<std::size_t>(p)]);
                }
        }
    }

    // ------------------------------------------------------------------
    // 3 and 4. AMR.
    // ------------------------------------------------------------------
    for (int pass = 1; pass <= max_level; ++pass) {
        std::fill(refine_mark.begin(), refine_mark.end(), 0);
        int requested = 0;

#pragma omp parallel for schedule(static) reduction(+:requested)
        for (int j = 0; j < fine_n; ++j) {
            for (int i = 0; i < fine_n; ++i) {
                    const int p = index2d(i, j, fine_n);
                    if (!active[static_cast<std::size_t>(p)]) {
                        continue;
                    }
                    if (level[static_cast<std::size_t>(p)] < max_level &&
                        importance[static_cast<std::size_t>(p)] >= 0.5) {
                        refine_mark[static_cast<std::size_t>(p)] = 1;
                        ++requested;
                    }
            }
        }

        if (requested == 0) {
            break;
        }

        for (int j = 0; j < fine_n; ++j) {
            for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (!refine_mark[static_cast<std::size_t>(p)]) {
                    continue;
                }

                const int parent_level = level[static_cast<std::size_t>(p)];
                const int w = cell_width(parent_level, max_level);
                const int half = w / 2;
                if (half < 1) {
                    continue;
                }

                const double parent_value = value[static_cast<std::size_t>(p)];
                const double parent_importance = importance[static_cast<std::size_t>(p)];
                const int child_level = parent_level + 1;

                const int child_i[4] = {i, i + half, i, i + half};
                const int child_j[4] = {j, j, j + half, j + half};
                for (int c = 0; c < 4; ++c) {
                    const int q = index2d(child_i[c], child_j[c], fine_n);
                    active[static_cast<std::size_t>(q)] = 1;
                    level[static_cast<std::size_t>(q)] = static_cast<unsigned char>(child_level);
                    value[static_cast<std::size_t>(q)] = parent_value;
                    next_value[static_cast<std::size_t>(q)] = parent_value;
                    importance[static_cast<std::size_t>(q)] = parent_importance;
                }
            }
        }

        int leaves = 0;
        for (int p = 0; p < array_size; ++p) {
            if (active[static_cast<std::size_t>(p)]) {
                ++leaves;
            }
        }

        std::cout << "pass " << pass
                  << ": requested=" << requested
                  << ", leaves=" << leaves << '\n';
    }

    // ------------------------------------------------------------------
    // Output. Scan the same 2D arrays and draw active entries only.
    // ------------------------------------------------------------------
    int leaves = 0;
    for (int p = 0; p < array_size; ++p) {
        if (active[static_cast<std::size_t>(p)]) {
            ++leaves;
        }
    }

    std::ofstream out(output);
    if (!out) {
        std::cerr << "failed to open output: " << output << '\n';
        return 1;
    }

    if (ends_with(output, ".vtk")) {
        out << "# vtk DataFile Version 3.0\n";
        out << "dense fine-grid AMR\n";
        out << "ASCII\n";
        out << "DATASET UNSTRUCTURED_GRID\n";
        out << "POINTS " << leaves * 4 << " double\n";
        out << std::setprecision(17);

        for (int j = 0; j < fine_n; ++j) {
            for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (!active[static_cast<std::size_t>(p)]) {
                    continue;
                }
                const int w = cell_width(level[static_cast<std::size_t>(p)], max_level);
                const double x0 = static_cast<double>(i) / static_cast<double>(fine_n);
                const double y0 = static_cast<double>(j) / static_cast<double>(fine_n);
                const double size = static_cast<double>(w) / static_cast<double>(fine_n);
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
    } else {
        const char *colors[] = {
            "#f8fafc", "#dbeafe", "#bbf7d0", "#fde68a",
            "#fca5a5", "#ddd6fe", "#bae6fd", "#fecdd3"};
        constexpr double canvas = 1000.0;
        constexpr double margin = 24.0;
        constexpr double scale = canvas - 2.0 * margin;

        out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1000 1000\">\n";
        out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
        out << std::fixed << std::setprecision(3);
        out << "<g stroke=\"#1f2937\" stroke-width=\"0.55\" vector-effect=\"non-scaling-stroke\">\n";

        for (int j = 0; j < fine_n; ++j) {
            for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (!active[static_cast<std::size_t>(p)]) {
                    continue;
                }
                const int lev = level[static_cast<std::size_t>(p)];
                const int w = cell_width(lev, max_level);
                const double x = margin + static_cast<double>(i) / static_cast<double>(fine_n) * scale;
                const double y = margin + (1.0 - static_cast<double>(j + w) / static_cast<double>(fine_n)) * scale;
                const double size = static_cast<double>(w) / static_cast<double>(fine_n) * scale;
                const int part = owner_part_from_fine_row(j, w, coarse_n, parts, fine_n);

                out << "<rect x=\"" << x
                    << "\" y=\"" << y
                    << "\" width=\"" << size
                    << "\" height=\"" << size
                    << "\" fill=\"" << colors[lev % 8]
                    << "\"><title>fine(" << i << ", " << j << ")"
                    << ", row_part " << part
                    << ", level " << lev
                    << ", value " << value[static_cast<std::size_t>(p)]
                    << "</title></rect>\n";
            }
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
    }

    std::cout << "wrote " << output << '\n';
    return 0;
}
