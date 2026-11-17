
#include <mpi.h>
#include <omp.h>
#include "life_common.hpp"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Params p = parse_args(argc, argv);
    int n = p.n;
    if (n < size) {   // каждому процессу нужна хотя бы одна строка
        if (rank == 0) std::cerr << "N must be >= number of processes\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> counts(size), starts(size);
    for (int r = 0, pos = 0; r < size; r++) {
        counts[r] = n / size + (r < n % size ? 1 : 0);
        starts[r] = pos;
        pos += counts[r];
    }
    int local_rows = counts[rank];
    int start = starts[rank];

    int up   = (rank - 1 + size) % size;
    int down = (rank + 1) % size;

    Grid cur((size_t)(local_rows + 2) * n, 0);
    Grid next((size_t)(local_rows + 2) * n, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= local_rows; i++)
        for (int j = 0; j < n; j++)
            at(cur, i, j, n) = initial_cell(start + i - 1, j);   // глобальная строка

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int s = 0; s < p.steps; s++) {
        MPI_Sendrecv(&at(cur, 1, 0, n),              n, MPI_UNSIGNED_CHAR, up,   0,
                     &at(cur, local_rows + 1, 0, n), n, MPI_UNSIGNED_CHAR, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&at(cur, local_rows, 0, n),     n, MPI_UNSIGNED_CHAR, down, 1,
                     &at(cur, 0, 0, n),              n, MPI_UNSIGNED_CHAR, up,   1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= local_rows; i++)
            for (int j = 0; j < n; j++)
                at(next, i, j, n) = life_rule(at(cur, i, j, n), count_neighbors(cur, i, j, n));

        cur.swap(next);
    }

    double local_time = MPI_Wtime() - t0;
    double time_max;   // время программы = время самого медленного процесса
    MPI_Reduce(&local_time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long local_alive = 0, alive = 0;
    for (int i = 1; i <= local_rows; i++)
        for (int j = 0; j < n; j++)
            local_alive += at(cur, i, j, n);
    MPI_Reduce(&local_alive, &alive, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!p.out.empty()) {
        std::vector<int> cnt_cells(size), disp_cells(size);
        for (int r = 0; r < size; r++) {
            cnt_cells[r] = counts[r] * n;
            disp_cells[r] = starts[r] * n;
        }
        Grid full;
        if (rank == 0) full.assign((size_t)(n + 2) * n, 0);
        MPI_Gatherv(&at(cur, 1, 0, n), local_rows * n, MPI_UNSIGNED_CHAR,
                    rank == 0 ? &at(full, 1, 0, n) : nullptr,
                    cnt_cells.data(), disp_cells.data(), MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);
        if (rank == 0) save_grid(p.out, full, n, n);
    }

    if (rank == 0) {
        int workers = size * omp_get_max_threads();   // процессы * потоки
        std::string ver = "mpi_p" + std::to_string(size) + "_t" + std::to_string(omp_get_max_threads());
        print_result(ver.c_str(), p, workers, time_max, alive);
    }

    MPI_Finalize();
    return 0;
}
