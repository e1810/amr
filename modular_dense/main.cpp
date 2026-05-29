#include "dense_amr.hpp"
#include "dense_amr_output.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <omp.h>

int main(int argc, char **argv) {
    const int coarse_n = argc > 1 ? std::atoi(argv[1]) : 16;
    const int parts = argc > 2 ? std::atoi(argv[2]) : 4;
    const int max_level = argc > 3 ? std::atoi(argv[3]) : 3;
    const double radius = argc > 4 ? std::atof(argv[4]) : 0.28;
    const std::string output = argc > 5 ? argv[5] : "amr_modular.svg";

    try {
        omp_set_dynamic(0);
        omp_set_num_threads(parts);

        DenseAmr amr = make_dense_amr(coarse_n, parts, max_level);
        std::cout << "OpenMP max threads: " << omp_get_max_threads()
                  << ", requested threads: " << parts << '\n';
        std::cout << "initial: coarse_grid=" << amr.coarse_n << "x" << amr.coarse_n
                  << ", fine_grid=" << amr.fine_n << "x" << amr.fine_n
                  << ", row_parts=" << amr.parts << '\n';
        for (int part = 0; part < amr.parts; ++part) {
            std::cout << "static part " << part << ": coarse rows ["
                      << row_begin(amr.coarse_n, amr.parts, part) << ", "
                      << row_end(amr.coarse_n, amr.parts, part) << ")\n";
        }

        initialize_circle_field(amr, radius);
        update_from_neighbors(amr, 0.02);
        compute_importance(amr);

        for (int pass = 1; pass <= max_level; ++pass) {
            const int requested = refine_once(amr, 0.5);
            if (requested == 0) {
                break;
            }
            std::cout << "pass " << pass
                      << ": requested=" << requested
                      << ", leaves=" << count_leaves(amr) << '\n';
        }

        write_output(amr, output);
        std::cout << "wrote " << output << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
