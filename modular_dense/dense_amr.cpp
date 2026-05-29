#include "dense_amr.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

DenseAmr make_dense_amr(int coarse_n, int parts, int max_level) {
    if (coarse_n < 1) {
        throw std::invalid_argument("coarse_n must be >= 1");
    }
    if (parts < 1 || parts > coarse_n) {
        throw std::invalid_argument("parts must be in [1, coarse_n]");
    }
    if (max_level < 0 || max_level > 10) {
        throw std::invalid_argument("max_level must be in [0, 10]");
    }

    DenseAmr amr;
    amr.coarse_n = coarse_n;
    amr.parts = parts;
    amr.max_level = max_level;
    amr.refine_factor = 1 << max_level;
    amr.fine_n = coarse_n * amr.refine_factor;
    amr.coarse_width = amr.refine_factor;
    amr.array_size = amr.fine_n * amr.fine_n;

    amr.active.assign(static_cast<std::size_t>(amr.array_size), 0);
    amr.level.assign(static_cast<std::size_t>(amr.array_size), 0);
    amr.refine_mark.assign(static_cast<std::size_t>(amr.array_size), 0);
    amr.owner.assign(static_cast<std::size_t>(amr.array_size), -1);
    amr.value.assign(static_cast<std::size_t>(amr.array_size), 0.0);
    amr.next_value.assign(static_cast<std::size_t>(amr.array_size), 0.0);
    amr.importance.assign(static_cast<std::size_t>(amr.array_size), 0.0);

    for (int coarse_j = 0; coarse_j < coarse_n; ++coarse_j) {
        for (int coarse_i = 0; coarse_i < coarse_n; ++coarse_i) {
            const int i = coarse_i * amr.coarse_width;
            const int j = coarse_j * amr.coarse_width;
            const int p = index2d(i, j, amr.fine_n);
            amr.active[static_cast<std::size_t>(p)] = 1;
            amr.level[static_cast<std::size_t>(p)] = 0;
        }
    }

    return amr;
}

void initialize_circle_field(DenseAmr &amr, double radius) {
#pragma omp parallel for schedule(static)
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
            const double x1 = x0 + size;
            const double y1 = y0 + size;
            const double nearest_x = std::max(x0, std::min(0.5, x1));
            const double nearest_y = std::max(y0, std::min(0.5, y1));
            const double dx = nearest_x - 0.5;
            const double dy = nearest_y - 0.5;
            amr.value[static_cast<std::size_t>(p)] = (dx * dx + dy * dy <= radius * radius) ? 1.0 : 0.0;
        }
    }
}

void rebuild_owner(DenseAmr &amr) {
    std::fill(amr.owner.begin(), amr.owner.end(), -1);

    for (int j = 0; j < amr.fine_n; ++j) {
        for (int i = 0; i < amr.fine_n; ++i) {
            const int p = index2d(i, j, amr.fine_n);
            if (!amr.active[static_cast<std::size_t>(p)]) {
                continue;
            }

            const int w = cell_width(amr.level[static_cast<std::size_t>(p)], amr.max_level);
            for (int jj = j; jj < j + w; ++jj) {
                for (int ii = i; ii < i + w; ++ii) {
                    amr.owner[static_cast<std::size_t>(index2d(ii, jj, amr.fine_n))] = p;
                }
            }
        }
    }
}

namespace {

double face_average(const DenseAmr &amr,
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

        if (direction == 0) {
            ni = i - 1;
            nj = j + s;
        } else if (direction == 1) {
            ni = i + w;
            nj = j + s;
        } else if (direction == 2) {
            ni = i + s;
            nj = j - 1;
        } else {
            ni = i + s;
            nj = j + w;
        }

        if (ni < 0 || ni >= amr.fine_n || nj < 0 || nj >= amr.fine_n) {
            continue;
        }

        const int q = amr.owner[static_cast<std::size_t>(index2d(ni, nj, amr.fine_n))];
        if (q < 0 || q == self) {
            continue;
        }

        sum += amr.value[static_cast<std::size_t>(q)];
        ++count;
    }

    return count == 0 ? fallback : sum / static_cast<double>(count);
}

}  // namespace

void diffuse_one_step(DenseAmr &amr, double diffusion, double dt) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < amr.fine_n; ++j) {
        for (int i = 0; i < amr.fine_n; ++i) {
            const int p = index2d(i, j, amr.fine_n);
            if (!amr.active[static_cast<std::size_t>(p)]) {
                continue;
            }

            const int w = cell_width(amr.level[static_cast<std::size_t>(p)], amr.max_level);
            const double center = amr.value[static_cast<std::size_t>(p)];
            const double left = face_average(amr, p, i, j, w, 0, center);
            const double right = face_average(amr, p, i, j, w, 1, center);
            const double bottom = face_average(amr, p, i, j, w, 2, center);
            const double top = face_average(amr, p, i, j, w, 3, center);
            const double dx = static_cast<double>(w) / static_cast<double>(amr.fine_n);
            const double laplacian = (left + right + bottom + top - 4.0 * center) / (dx * dx);
            amr.next_value[static_cast<std::size_t>(p)] = center + dt * diffusion * laplacian;
        }
    }

#pragma omp parallel for schedule(static)
    for (int j = 0; j < amr.fine_n; ++j) {
        for (int i = 0; i < amr.fine_n; ++i) {
            const int p = index2d(i, j, amr.fine_n);
            if (amr.active[static_cast<std::size_t>(p)]) {
                amr.value[static_cast<std::size_t>(p)] = amr.next_value[static_cast<std::size_t>(p)];
            }
        }
    }
}

void compute_gradient_importance(DenseAmr &amr) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < amr.fine_n; ++j) {
        for (int i = 0; i < amr.fine_n; ++i) {
            const int p = index2d(i, j, amr.fine_n);
            if (!amr.active[static_cast<std::size_t>(p)]) {
                continue;
            }

            const int w = cell_width(amr.level[static_cast<std::size_t>(p)], amr.max_level);
            const double center = amr.value[static_cast<std::size_t>(p)];
            const double left = face_average(amr, p, i, j, w, 0, center);
            const double right = face_average(amr, p, i, j, w, 1, center);
            const double bottom = face_average(amr, p, i, j, w, 2, center);
            const double top = face_average(amr, p, i, j, w, 3, center);
            const double jump_x = std::max(std::abs(right - center), std::abs(left - center));
            const double jump_y = std::max(std::abs(top - center), std::abs(bottom - center));
            amr.importance[static_cast<std::size_t>(p)] = std::max(jump_x, jump_y);
        }
    }
}

int refine_once(DenseAmr &amr, double threshold) {
    std::fill(amr.refine_mark.begin(), amr.refine_mark.end(), 0);
    int requested = 0;

#pragma omp parallel for schedule(static) reduction(+:requested)
    for (int j = 0; j < amr.fine_n; ++j) {
        for (int i = 0; i < amr.fine_n; ++i) {
            const int p = index2d(i, j, amr.fine_n);
            if (!amr.active[static_cast<std::size_t>(p)]) {
                continue;
            }
            if (amr.level[static_cast<std::size_t>(p)] < amr.max_level &&
                amr.importance[static_cast<std::size_t>(p)] >= threshold) {
                amr.refine_mark[static_cast<std::size_t>(p)] = 1;
                ++requested;
            }
        }
    }

    for (int j = 0; j < amr.fine_n; ++j) {
        for (int i = 0; i < amr.fine_n; ++i) {
            const int p = index2d(i, j, amr.fine_n);
            if (!amr.refine_mark[static_cast<std::size_t>(p)]) {
                continue;
            }

            const int parent_level = amr.level[static_cast<std::size_t>(p)];
            const int w = cell_width(parent_level, amr.max_level);
            const int half = w / 2;
            if (half < 1) {
                continue;
            }

            const double parent_value = amr.value[static_cast<std::size_t>(p)];
            const double parent_importance = amr.importance[static_cast<std::size_t>(p)];
            const int child_level = parent_level + 1;
            const int child_i[4] = {i, i + half, i, i + half};
            const int child_j[4] = {j, j, j + half, j + half};

            for (int c = 0; c < 4; ++c) {
                const int q = index2d(child_i[c], child_j[c], amr.fine_n);
                amr.active[static_cast<std::size_t>(q)] = 1;
                amr.level[static_cast<std::size_t>(q)] = static_cast<unsigned char>(child_level);
                amr.value[static_cast<std::size_t>(q)] = parent_value;
                amr.next_value[static_cast<std::size_t>(q)] = parent_value;
                amr.importance[static_cast<std::size_t>(q)] = parent_importance;
            }
        }
    }

    return requested;
}

int count_leaves(const DenseAmr &amr) {
    int leaves = 0;
    for (unsigned char is_active : amr.active) {
        if (is_active) {
            ++leaves;
        }
    }
    return leaves;
}

double heat_mass(const DenseAmr &amr) {
    double mass = 0.0;
    for (int j = 0; j < amr.fine_n; ++j) {
        for (int i = 0; i < amr.fine_n; ++i) {
            const int p = index2d(i, j, amr.fine_n);
            if (!amr.active[static_cast<std::size_t>(p)]) {
                continue;
            }
            const int w = cell_width(amr.level[static_cast<std::size_t>(p)], amr.max_level);
            const double cell_size = static_cast<double>(w) / static_cast<double>(amr.fine_n);
            mass += amr.value[static_cast<std::size_t>(p)] * cell_size * cell_size;
        }
    }
    return mass;
}
