
#include <algorithm>
#include <omp.h>
#include "life_common.hpp"

int main(int argc, char **argv)
{
    Params p = parse_args(argc, argv);
    int n = p.n;

    Grid cur((size_t)(n + 2) * n, 0);
    Grid next((size_t)(n + 2) * n, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= n; i++)
        for (int j = 0; j < n; j++)
            at(cur, i, j, n) = initial_cell(i - 1, j);

    double t0 = omp_get_wtime();
    for (int s = 0; s < p.steps; s++) {
        std::copy_n(&at(cur, n, 0, n), n, &at(cur, 0, 0, n));
        std::copy_n(&at(cur, 1, 0, n), n, &at(cur, n + 1, 0, n));

        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= n; i++)
            for (int j = 0; j < n; j++)
                at(next, i, j, n) = life_rule(at(cur, i, j, n), count_neighbors(cur, i, j, n));

        cur.swap(next);
    }
    double t1 = omp_get_wtime();

    long long alive = 0;
    #pragma omp parallel for reduction(+:alive)
    for (int i = 1; i <= n; i++)
        for (int j = 0; j < n; j++)
            alive += at(cur, i, j, n);

    print_result("openmp", p, omp_get_max_threads(), t1 - t0, alive);
    if (!p.out.empty()) save_grid(p.out, cur, n, n);
    return 0;
}
