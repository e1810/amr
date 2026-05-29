#include "dense_amr.hpp"
#include "dense_amr_output.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <omp.h>

int main(int argc, char **argv) {
    const int coarse_n = argc > 1 ? std::atoi(argv[1]) : 16;
    const int parts = argc > 2 ? std::atoi(argv[2]) : 4;
    const int max_level = argc > 3 ? std::atoi(argv[3]) : 3;
    const double radius = argc > 4 ? std::atof(argv[4]) : 0.28;
    const int steps = argc > 5 ? std::atoi(argv[5]) : 100;
    const double diffusion = argc > 6 ? std::atof(argv[6]) : 0.01;
    const std::string output = argc > 7 ? argv[7] : "amr_modular.svg";

    try {
        if (radius <= 0.0 || steps < 0 || diffusion <= 0.0) {
            throw std::invalid_argument(
                "usage: ./amr_dense_modular [coarse_n>=1] [parts in 1..coarse_n] "
                "[max_level 0..10] [radius>0] [steps>=0] [diffusion>0] [output.svg|output.vtk]");
        }

        omp_set_dynamic(0);
        omp_set_num_threads(parts);

        DenseAmr amr = make_dense_amr(coarse_n, parts, max_level);
        const double dx_min = 1.0 / static_cast<double>(amr.fine_n);
        const double dt = 0.20 * dx_min * dx_min / diffusion;
        const double refine_threshold = 0.05;

        std::cout << "OpenMP max threads: " << omp_get_max_threads()
                  << ", requested threads: " << parts << '\n';
        std::cout << "initial: coarse_grid=" << amr.coarse_n << "x" << amr.coarse_n
                  << ", fine_grid=" << amr.fine_n << "x" << amr.fine_n
                  << ", row_parts=" << amr.parts << '\n';
        std::cout << "heat diffusion: steps=" << steps
                  << ", diffusion=" << diffusion
                  << ", dt=" << dt
                  << ", refine_threshold=" << refine_threshold << '\n';
        for (int part = 0; part < amr.parts; ++part) {
            std::cout << "static part " << part << ": coarse rows ["
                      << row_begin(amr.coarse_n, amr.parts, part) << ", "
                      << row_end(amr.coarse_n, amr.parts, part) << ")\n";
        }

        initialize_circle_field(amr, radius);
        const double initial_heat = heat_mass(amr);

        for (int step = 1; step <= steps; ++step) {
            rebuild_owner(amr);
            diffuse_one_step(amr, diffusion, dt);
            compute_gradient_importance(amr);

            const int requested = refine_once(amr, refine_threshold);
            if (requested > 0) {
                std::cout << "step " << step
                          << ": refined=" << requested
                          << ", leaves=" << count_leaves(amr) << '\n';
            }
        }

        const double final_heat = heat_mass(amr);
        std::cout << "heat: initial=" << initial_heat
                  << ", final=" << final_heat
                  << ", difference=" << final_heat - initial_heat << '\n';

        write_output(amr, output);
        std::cout << "wrote " << output << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
