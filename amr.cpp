#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <omp.h>

// Dense-row AMR heat-diffusion prototype.
//
// Mesh representation:
//   - The storage is a dense fine_n x fine_n set of arrays.
//   - active[p] means that fine-grid location p is the lower-left anchor of
//     a current AMR leaf cell.
//   - level[p] gives the refinement level of that leaf.
//   - owner[q] is rebuilt from active/level and maps each finest-grid point
//     to the active leaf that covers it.
//
// Parallelism:
//   - OpenMP regions scan fine-grid rows with schedule(static).
//   - Field updates write one array entry per active leaf, so they have no
//     write-write races.
//   - Mesh topology changes, refine and coarsen, are applied sequentially
//     after parallel marking so active/level stay consistent.

int index2d(int i, int j, int n) {
    return j * n + i;
}

std::size_t state_index(int cell, int component, int state_components) {
    return static_cast<std::size_t>(cell) * static_cast<std::size_t>(state_components) +
           static_cast<std::size_t>(component);
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

std::string snapshot_path(const std::string &prefix, int step) {
    std::ostringstream path;
    path << prefix << '_' << std::setw(6) << std::setfill('0') << step << ".txt";
    return path.str();
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

void write_snapshot_text(const std::string &path,
                         int step,
                         double time,
                         int coarse_n,
                         int parts,
                         int max_level,
                         int fine_n,
                         const std::string &initial_pattern,
                         double initial_scale,
                         int state_components,
                         const std::vector<unsigned char> &active,
                         const std::vector<unsigned char> &level,
                         const std::vector<double> &value,
                         const std::vector<double> &importance) {
    int leaves = 0;
    for (unsigned char is_active : active) {
        if (is_active) {
            ++leaves;
        }
    }

    prepare_parent_directory(path);
    std::ofstream out(path);
    if (!out) {
        std::cerr << "failed to open snapshot: " << path << '\n';
        return;
    }

    out << "# amr_dense_snapshot_v1\n";
    out << "# step " << step << "\n";
    out << "# time " << std::setprecision(17) << time << "\n";
    out << "# coarse_n " << coarse_n << "\n";
    out << "# parts " << parts << "\n";
    out << "# max_level " << max_level << "\n";
    out << "# fine_n " << fine_n << "\n";
    out << "# initial_pattern " << initial_pattern << "\n";
    out << "# initial_scale " << std::setprecision(17) << initial_scale << "\n";
    out << "# state_components " << state_components << "\n";
    out << "# leaves " << leaves << "\n";
    out << "# columns i j level value importance\n";

    out << std::setprecision(17);
    for (int j = 0; j < fine_n; ++j) {
        for (int i = 0; i < fine_n; ++i) {
            const int p = index2d(i, j, fine_n);
            if (!active[static_cast<std::size_t>(p)]) {
                continue;
            }
            out << i << ' '
                << j << ' '
                << static_cast<int>(level[static_cast<std::size_t>(p)]) << ' '
                << value[state_index(p, 0, state_components)] << ' '
                << importance[static_cast<std::size_t>(p)] << '\n';
        }
    }
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
                    int state_components,
                    int component,
                    int self,
                    int i,
                    int j,
                    int w,
                    int direction,
                    double fallback) {
    double sum = 0.0;
    int count = 0;

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
        if (q < 0 || q == self) {
            continue;
        }

        sum += value[state_index(q, component, state_components)];
        ++count;
    }

    return count == 0 ? fallback : sum / static_cast<double>(count);
}

int main(int argc, char **argv) {
    // ------------------------------------------------------------------
    // Input parameters.
    //
    // coarse_n:      number of cells per side in the initial mesh.
    // max_level:     maximum AMR depth; fine_n = coarse_n * 2^max_level.
    // parts:         OpenMP thread count and row-part guide for logs.
    // initial_scale:    size parameter for checker/fine_checker/stripe/square/circle/hotspot/multi_circle data.
    // steps:            number of explicit heat-diffusion time steps.
    // state_components: number of per-cell state variables advanced by the stencil.
    // ------------------------------------------------------------------
    const int coarse_n = argc > 1 ? std::atoi(argv[1]) : 16;
    const int parts = argc > 2 ? std::atoi(argv[2]) : 4;
    const int max_level = argc > 3 ? std::atoi(argv[3]) : 3;
    const double initial_scale = argc > 4 ? std::atof(argv[4]) : 0.25;
    const int steps = argc > 5 ? std::atoi(argv[5]) : 100;
    const double diffusion = argc > 6 ? std::atof(argv[6]) : 0.01;
    const int snapshot_interval = argc > 7 ? std::atoi(argv[7]) : 0;
    const std::string snapshot_prefix = argc > 8 ? argv[8] : "snapshots/amr_step";
    const std::string initial_pattern = argc > 9 ? argv[9] : "checker";
    const int diffusion_substeps = argc > 10 ? std::atoi(argv[10]) : 1;
    const int state_components = argc > 11 ? std::atoi(argv[11]) : 1;
    const bool known_initial_pattern =
        initial_pattern == "checker" ||
        initial_pattern == "fine_checker" ||
        initial_pattern == "stripe" ||
        initial_pattern == "square" ||
        initial_pattern == "circle" ||
        initial_pattern == "hotspot" ||
        initial_pattern == "multi_circle";

    if (coarse_n < 1 || parts < 1 || parts > coarse_n || max_level < 0 || max_level > 10 ||
        initial_scale <= 0.0 || steps < 0 || diffusion <= 0.0 || snapshot_interval < 0 ||
        diffusion_substeps < 1 || state_components < 1 ||
        !known_initial_pattern) {
        std::cerr << "usage: " << argv[0]
                  << " [coarse_n>=1] [parts in 1..coarse_n] [max_level 0..10]"
                  << " [initial_scale>0] [steps>=0] [diffusion>0]"
                  << " [snapshot_interval>=0] [snapshot_prefix]"
                  << " [initial_pattern: checker|fine_checker|stripe|square|circle|hotspot|multi_circle]"
                  << " [diffusion_substeps>=1] [state_components>=1]\n";
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
    const std::size_t state_size = static_cast<std::size_t>(array_size) *
                                   static_cast<std::size_t>(state_components);
    const double dx_min = 1.0 / static_cast<double>(fine_n);
    const double dt = 0.20 * dx_min * dx_min / diffusion;
    const double output_dt = dt * static_cast<double>(diffusion_substeps);
    const double refine_threshold = 0.05;
    const double coarsen_threshold = 0.015;
    const double coarsen_value_tolerance = 0.03;
    const int amr_log_interval = snapshot_interval > 0 ? snapshot_interval : 100;

    // Dense storage for AMR leaves and per-cell state fields.
    // Only active[p] entries are physical cells; inactive entries are just
    // unused slots on the finest-grid canvas.
    std::vector<unsigned char> active(static_cast<std::size_t>(array_size), 0);
    std::vector<unsigned char> level(static_cast<std::size_t>(array_size), 0);
    std::vector<unsigned char> refine_mark(static_cast<std::size_t>(array_size), 0);
    std::vector<unsigned char> coarsen_mark(static_cast<std::size_t>(array_size), 0);
    std::vector<int> owner(static_cast<std::size_t>(array_size), -1);
    std::vector<double> value(state_size, 0.0);
    std::vector<double> next_value(state_size, 0.0);
    std::vector<double> importance(static_cast<std::size_t>(array_size), 0.0);

    // ------------------------------------------------------------------
    // Build the initial level-0 mesh.
    //
    // A leaf cell is stored at its lower-left anchor on the fine-grid canvas.
    // At level 0, neighboring anchors are separated by coarse_width.
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
    std::cout << "heat diffusion: steps=" << steps
              << ", diffusion=" << diffusion
              << ", dt=" << dt
              << ", diffusion_substeps=" << diffusion_substeps
              << ", state_components=" << state_components
              << ", output_dt=" << output_dt
              << ", refine_threshold=" << refine_threshold
              << ", coarsen_threshold=" << coarsen_threshold
              << ", coarsen_value_tolerance=" << coarsen_value_tolerance << '\n';
    std::cout << "initial condition: pattern=" << initial_pattern
              << ", scale=" << initial_scale << '\n';
    if (snapshot_interval > 0) {
        std::cout << "snapshots: every " << snapshot_interval
                  << " steps, prefix=" << snapshot_prefix << '\n';
    }
    for (int part = 0; part < parts; ++part) {
        std::cout << "static part " << part << ": coarse rows ["
                  << row_begin(coarse_n, parts, part) << ", "
                  << row_end(coarse_n, parts, part) << ")\n";
    }

    // ------------------------------------------------------------------
    // Parallel region 1: initialize the scalar field.
    //
    // Each iteration reads active/level and writes all state components for one
    // active leaf. No two iterations write the same p.
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
                const double x = 0.5 * (x0 + x1);
                const double y = 0.5 * (y0 + y1);

                double primary_value = 0.0;
                if (initial_pattern == "checker" || initial_pattern == "fine_checker") {
                    const int blocks = std::max(2, static_cast<int>(std::round(1.0 / initial_scale)));
                    const int block_i = std::min(blocks - 1, static_cast<int>(x * blocks));
                    const int block_j = std::min(blocks - 1, static_cast<int>(y * blocks));
                    primary_value = ((block_i + block_j) % 2 == 0) ? 1.0 : 0.0;
                } else if (initial_pattern == "stripe") {
                    primary_value = (std::abs(x - 0.5) <= 0.5 * initial_scale) ? 1.0 : 0.0;
                } else if (initial_pattern == "square") {
                    primary_value =
                        (std::abs(x - 0.5) <= 0.5 * initial_scale &&
                         std::abs(y - 0.5) <= 0.5 * initial_scale) ? 1.0 : 0.0;
                } else if (initial_pattern == "hotspot") {
                    const double hotspot_x = 0.35;
                    const double hotspot_y = 0.35;
                    const double nearest_x = std::max(x0, std::min(hotspot_x, x1));
                    const double nearest_y = std::max(y0, std::min(hotspot_y, y1));
                    const double dx = nearest_x - hotspot_x;
                    const double dy = nearest_y - hotspot_y;
                    primary_value = (dx * dx + dy * dy <= initial_scale * initial_scale) ? 1.0 : 0.0;
                } else if (initial_pattern == "multi_circle") {
                    const double centers[4][2] = {
                        {0.25, 0.25},
                        {0.75, 0.25},
                        {0.25, 0.75},
                        {0.75, 0.75},
                    };
                    bool inside = false;
                    for (const auto &center : centers) {
                        const double nearest_x = std::max(x0, std::min(center[0], x1));
                        const double nearest_y = std::max(y0, std::min(center[1], y1));
                        const double dx = nearest_x - center[0];
                        const double dy = nearest_y - center[1];
                        if (dx * dx + dy * dy <= initial_scale * initial_scale) {
                            inside = true;
                            break;
                        }
                    }
                    primary_value = inside ? 1.0 : 0.0;
                } else {
                    const double nearest_x = std::max(x0, std::min(0.5, x1));
                    const double nearest_y = std::max(y0, std::min(0.5, y1));
                    const double dx = nearest_x - 0.5;
                    const double dy = nearest_y - 0.5;
                    primary_value = (dx * dx + dy * dy <= initial_scale * initial_scale) ? 1.0 : 0.0;
                }

                value[state_index(p, 0, state_components)] = primary_value;
                for (int component = 1; component < state_components; ++component) {
                    const double wave = 0.05 *
                        std::sin(static_cast<double>(component + 1) * 3.14159265358979323846 * x) *
                        std::cos(static_cast<double>(component + 1) * 3.14159265358979323846 * y);
                    value[state_index(p, component, state_components)] =
                        primary_value * (1.0 + 0.01 * static_cast<double>(component)) + wave;
                }
        }
    }

    // Optional baseline snapshot before the first time step.
    if (snapshot_interval > 0) {
        write_snapshot_text(snapshot_path(snapshot_prefix, 0),
                            0,
                            0.0,
                            coarse_n,
                            parts,
                            max_level,
                            fine_n,
                            initial_pattern,
                            initial_scale,
                            state_components,
                            active,
                            level,
                            value,
                            importance);
    }

    // Integrated heat is a simple conservation check for the zero-flux run.
    double initial_heat = 0.0;
    for (int j = 0; j < fine_n; ++j) {
        for (int i = 0; i < fine_n; ++i) {
            const int p = index2d(i, j, fine_n);
            if (!active[static_cast<std::size_t>(p)]) {
                continue;
            }
            const int w = cell_width(level[static_cast<std::size_t>(p)], max_level);
            const double cell_size = static_cast<double>(w) / static_cast<double>(fine_n);
            initial_heat += value[state_index(p, 0, state_components)] * cell_size * cell_size;
        }
    }

    // ------------------------------------------------------------------
    // Time loop.
    //
    // Per step:
    //   1. Rebuild owner[] from the current AMR leaves.
    //   2. Advance heat diffusion on the current mesh.
    //   3. Measure importance from neighbor jumps.
    //   4. Mark cells for refine/coarsen in parallel.
    //   5. Apply coarsen/refine sequentially.
    // ------------------------------------------------------------------
    for (int step = 1; step <= steps; ++step) {
        // owner[] is a read-only lookup table for the parallel physics loops
        // below. It must be rebuilt after any mesh topology change.
        rebuild_owner(active, level, owner, fine_n, max_level);

        // Parallel region 2: advance heat diffusion for all state components.
        //
        // A finer fine_n forces a smaller stable dt. diffusion_substeps repeats
        // that stable update several times inside one visible step, so a heavy
        // grid can cover roughly the same physical time as the coarse example.
        // The mesh topology is fixed during these substeps; AMR decisions are
        // still made once per visible step below.
#pragma omp parallel
        {
            for (int substep = 0; substep < diffusion_substeps; ++substep) {
#pragma omp for schedule(static)
                for (int j = 0; j < fine_n; ++j) {
                    for (int i = 0; i < fine_n; ++i) {
                        const int p = index2d(i, j, fine_n);
                        if (!active[static_cast<std::size_t>(p)]) {
                            continue;
                        }

                        const int w = cell_width(level[static_cast<std::size_t>(p)], max_level);
                        const double dx = static_cast<double>(w) / static_cast<double>(fine_n);
                        for (int component = 0; component < state_components; ++component) {
                            const double center = value[state_index(p, component, state_components)];
                            const double left = face_average(owner, value, fine_n, state_components, component, p, i, j, w, 0, center);
                            const double right = face_average(owner, value, fine_n, state_components, component, p, i, j, w, 1, center);
                            const double bottom = face_average(owner, value, fine_n, state_components, component, p, i, j, w, 2, center);
                            const double top = face_average(owner, value, fine_n, state_components, component, p, i, j, w, 3, center);
                            const double laplacian = (left + right + bottom + top - 4.0 * center) / (dx * dx);
                            next_value[state_index(p, component, state_components)] =
                                center + dt * diffusion * laplacian;
                        }
                    }
                }

#pragma omp for schedule(static)
                for (int j = 0; j < fine_n; ++j) {
                    for (int i = 0; i < fine_n; ++i) {
                        const int p = index2d(i, j, fine_n);
                        if (active[static_cast<std::size_t>(p)]) {
                            for (int component = 0; component < state_components; ++component) {
                                value[state_index(p, component, state_components)] =
                                    next_value[state_index(p, component, state_components)];
                            }
                        }
                    }
                }
            }
        }

        // Parallel region 3: compute refinement importance.
        //
        // Importance is the largest jump between a cell and its four face
        // neighbors. Large jumps request refinement; small jumps may later
        // allow a complete sibling group to coarsen.
#pragma omp parallel for schedule(static)
        for (int j = 0; j < fine_n; ++j) {
            for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (!active[static_cast<std::size_t>(p)]) {
                    continue;
                }

                const int w = cell_width(level[static_cast<std::size_t>(p)], max_level);
                const double center = value[state_index(p, 0, state_components)];
                const double left = face_average(owner, value, fine_n, state_components, 0, p, i, j, w, 0, center);
                const double right = face_average(owner, value, fine_n, state_components, 0, p, i, j, w, 1, center);
                const double bottom = face_average(owner, value, fine_n, state_components, 0, p, i, j, w, 2, center);
                const double top = face_average(owner, value, fine_n, state_components, 0, p, i, j, w, 3, center);
                const double jump_x = std::max(std::abs(right - center), std::abs(left - center));
                const double jump_y = std::max(std::abs(top - center), std::abs(bottom - center));
                importance[static_cast<std::size_t>(p)] = std::max(jump_x, jump_y);
            }
        }

        // Parallel region 4: mark leaves that need refinement.
        //
        // The reduction counts requests for logging. Each iteration writes
        // refine_mark[p] only for its own active leaf.
        std::fill(refine_mark.begin(), refine_mark.end(), 0);
        std::fill(coarsen_mark.begin(), coarsen_mark.end(), 0);
        int requested = 0;
        int coarsen_requested = 0;

#pragma omp parallel for schedule(static) reduction(+:requested)
        for (int j = 0; j < fine_n; ++j) {
            for (int i = 0; i < fine_n; ++i) {
                const int p = index2d(i, j, fine_n);
                if (!active[static_cast<std::size_t>(p)]) {
                    continue;
                }
                if (level[static_cast<std::size_t>(p)] < max_level &&
                    importance[static_cast<std::size_t>(p)] >= refine_threshold) {
                    refine_mark[static_cast<std::size_t>(p)] = 1;
                    ++requested;
                }
            }
        }

        // Parallel region 5: mark complete sibling groups for coarsening.
        //
        // A group can coarsen only when all four children are active, at the
        // same level, not marked for refinement, and already smooth enough.
        // The lower-left child has the same fine-grid anchor as the parent,
        // so the sequential coarsen pass can reuse that slot.
#pragma omp parallel for schedule(static) reduction(+:coarsen_requested)
        for (int j = 0; j < fine_n; ++j) {
            for (int i = 0; i < fine_n; ++i) {
                const int p00 = index2d(i, j, fine_n);
                if (!active[static_cast<std::size_t>(p00)]) {
                    continue;
                }

                const int child_level = level[static_cast<std::size_t>(p00)];
                if (child_level == 0) {
                    continue;
                }

                const int child_w = cell_width(child_level, max_level);
                const int parent_w = 2 * child_w;
                if (i % parent_w != 0 || j % parent_w != 0 ||
                    i + child_w >= fine_n || j + child_w >= fine_n) {
                    continue;
                }

                const int p10 = index2d(i + child_w, j, fine_n);
                const int p01 = index2d(i, j + child_w, fine_n);
                const int p11 = index2d(i + child_w, j + child_w, fine_n);
                const int child[4] = {p00, p10, p01, p11};

                bool can_coarsen = true;
                double min_value = value[state_index(p00, 0, state_components)];
                double max_value = min_value;
                for (int c = 0; c < 4; ++c) {
                    const int q = child[c];
                    if (!active[static_cast<std::size_t>(q)] ||
                        level[static_cast<std::size_t>(q)] != child_level ||
                        refine_mark[static_cast<std::size_t>(q)] ||
                        importance[static_cast<std::size_t>(q)] >= coarsen_threshold) {
                        can_coarsen = false;
                        break;
                    }

                    min_value = std::min(min_value, value[state_index(q, 0, state_components)]);
                    max_value = std::max(max_value, value[state_index(q, 0, state_components)]);
                }

                if (can_coarsen && max_value - min_value <= coarsen_value_tolerance) {
                    coarsen_mark[static_cast<std::size_t>(p00)] = 1;
                    ++coarsen_requested;
                }
            }
        }

        // Mesh topology update 1: apply coarsening sequentially.
        //
        // Topology edits touch multiple array entries per parent, so keeping
        // this pass serial makes the data movement easy to inspect. The
        // parent's state is the component-wise average of the four child states.
        int coarsened = 0;
        if (coarsen_requested > 0) {
            for (int j = 0; j < fine_n; ++j) {
                for (int i = 0; i < fine_n; ++i) {
                    const int p00 = index2d(i, j, fine_n);
                    if (!coarsen_mark[static_cast<std::size_t>(p00)]) {
                        continue;
                    }

                    const int child_level = level[static_cast<std::size_t>(p00)];
                    const int child_w = cell_width(child_level, max_level);
                    const int p10 = index2d(i + child_w, j, fine_n);
                    const int p01 = index2d(i, j + child_w, fine_n);
                    const int p11 = index2d(i + child_w, j + child_w, fine_n);
                    const int child[4] = {p00, p10, p01, p11};

                    std::vector<double> parent_value(static_cast<std::size_t>(state_components), 0.0);
                    double parent_importance = 0.0;
                    for (int child_index = 0; child_index < 4; ++child_index) {
                        const int q = child[child_index];
                        for (int component = 0; component < state_components; ++component) {
                            parent_value[static_cast<std::size_t>(component)] +=
                                0.25 * value[state_index(q, component, state_components)];
                        }
                        parent_importance = std::max(parent_importance,
                                                     importance[static_cast<std::size_t>(q)]);
                    }

                    active[static_cast<std::size_t>(p00)] = 1;
                    level[static_cast<std::size_t>(p00)] = static_cast<unsigned char>(child_level - 1);
                    for (int component = 0; component < state_components; ++component) {
                        value[state_index(p00, component, state_components)] =
                            parent_value[static_cast<std::size_t>(component)];
                        next_value[state_index(p00, component, state_components)] =
                            parent_value[static_cast<std::size_t>(component)];
                    }
                    importance[static_cast<std::size_t>(p00)] = parent_importance;
                    refine_mark[static_cast<std::size_t>(p00)] = 0;

                    for (int child_index = 1; child_index < 4; ++child_index) {
                        const int q = child[child_index];
                        active[static_cast<std::size_t>(q)] = 0;
                        level[static_cast<std::size_t>(q)] = 0;
                        refine_mark[static_cast<std::size_t>(q)] = 0;
                        for (int component = 0; component < state_components; ++component) {
                            value[state_index(q, component, state_components)] = 0.0;
                            next_value[state_index(q, component, state_components)] = 0.0;
                        }
                        importance[static_cast<std::size_t>(q)] = 0.0;
                    }
                    ++coarsened;
                }
            }
        }

        // Mesh topology update 2: apply refinement sequentially.
        //
        // A marked parent is replaced by four child leaves. The scalar value
        // is copied into all children, preserving the parent cell average.
        int refined = 0;
        if (requested > 0) {
            for (int j = 0; j < fine_n; ++j) {
                for (int i = 0; i < fine_n; ++i) {
                    const int p = index2d(i, j, fine_n);
                    if (!refine_mark[static_cast<std::size_t>(p)] ||
                        !active[static_cast<std::size_t>(p)]) {
                        continue;
                    }

                    const int parent_level = level[static_cast<std::size_t>(p)];
                    const int w = cell_width(parent_level, max_level);
                    const int half = w / 2;
                    if (half < 1) {
                        continue;
                    }

                    std::vector<double> parent_value(static_cast<std::size_t>(state_components), 0.0);
                    for (int component = 0; component < state_components; ++component) {
                        parent_value[static_cast<std::size_t>(component)] =
                            value[state_index(p, component, state_components)];
                    }
                    const double parent_importance = importance[static_cast<std::size_t>(p)];
                    const int child_level = parent_level + 1;

                    const int child_i[4] = {i, i + half, i, i + half};
                    const int child_j[4] = {j, j, j + half, j + half};
                    for (int child_index = 0; child_index < 4; ++child_index) {
                        const int q = index2d(child_i[child_index], child_j[child_index], fine_n);
                        active[static_cast<std::size_t>(q)] = 1;
                        level[static_cast<std::size_t>(q)] = static_cast<unsigned char>(child_level);
                        for (int component = 0; component < state_components; ++component) {
                            value[state_index(q, component, state_components)] =
                                parent_value[static_cast<std::size_t>(component)];
                            next_value[state_index(q, component, state_components)] =
                                parent_value[static_cast<std::size_t>(component)];
                        }
                        importance[static_cast<std::size_t>(q)] = parent_importance;
                    }
                    ++refined;
                }
            }
        }

        if ((refined > 0 || coarsened > 0) &&
            (refined > 0 || step % amr_log_interval == 0 || step == steps)) {
            int leaves = 0;
            for (int p = 0; p < array_size; ++p) {
                if (active[static_cast<std::size_t>(p)]) {
                    ++leaves;
                }
            }

            std::cout << "step " << step
                      << ": refined=" << refined
                      << ", coarsened=" << coarsened
                      << ", leaves=" << leaves << '\n';
        }

        // Optional text snapshot. Rendering is intentionally a separate
        // program so visualization can change without rerunning the solver.
        if (snapshot_interval > 0 &&
            (step % snapshot_interval == 0 || step == steps)) {
            write_snapshot_text(snapshot_path(snapshot_prefix, step),
                                step,
                                step * output_dt,
                                coarse_n,
                                parts,
                                max_level,
                                fine_n,
                                initial_pattern,
                                initial_scale,
                                state_components,
                                active,
                                level,
                                value,
                                importance);
        }
    }

    // ------------------------------------------------------------------
    // Final summary. Visualization is handled by render_amr_snapshot using
    // the text snapshots written above.
    // ------------------------------------------------------------------

    // Recompute integrated heat after the final mesh update.
    double final_heat = 0.0;
    for (int j = 0; j < fine_n; ++j) {
        for (int i = 0; i < fine_n; ++i) {
            const int p = index2d(i, j, fine_n);
            if (!active[static_cast<std::size_t>(p)]) {
                continue;
            }
            const int w = cell_width(level[static_cast<std::size_t>(p)], max_level);
            const double cell_size = static_cast<double>(w) / static_cast<double>(fine_n);
            final_heat += value[state_index(p, 0, state_components)] * cell_size * cell_size;
        }
    }
    std::cout << "heat: initial=" << initial_heat
              << ", final=" << final_heat
              << ", difference=" << final_heat - initial_heat << '\n';

    return 0;
}
