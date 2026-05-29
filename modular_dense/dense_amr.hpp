#pragma once

#include <string>
#include <vector>

struct DenseAmr {
    int coarse_n = 0;
    int parts = 1;
    int max_level = 0;
    int refine_factor = 1;
    int fine_n = 0;
    int coarse_width = 1;
    int array_size = 0;

    std::vector<unsigned char> active;
    std::vector<unsigned char> level;
    std::vector<unsigned char> refine_mark;
    std::vector<int> owner;
    std::vector<double> value;
    std::vector<double> next_value;
    std::vector<double> importance;
};

int index2d(int i, int j, int n);
int row_begin(int coarse_n, int parts, int part);
int row_end(int coarse_n, int parts, int part);
int cell_width(int level, int max_level);
int owner_part_from_fine_row(int fine_row, int width, int coarse_n, int parts, int fine_n);

DenseAmr make_dense_amr(int coarse_n, int parts, int max_level);
void initialize_circle_field(DenseAmr &amr, double radius);
void rebuild_owner(DenseAmr &amr);
void update_from_neighbors(DenseAmr &amr, double weight);
void compute_importance(DenseAmr &amr);
int refine_once(DenseAmr &amr, double threshold);
int count_leaves(const DenseAmr &amr);
