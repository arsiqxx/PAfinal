
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Cell = std::uint8_t;            // 0 = мёртвая, 1 = живая
using Grid = std::vector<Cell>;

// Доступ к клетке (i, j) в поле шириной n.
inline Cell &at(Grid &g, int i, int j, int n) { return g[(size_t)i * n + j]; }
inline Cell at(const Grid &g, int i, int j, int n) { return g[(size_t)i * n + j]; }

// Начальное значение клетки зависит ТОЛЬКО от её глобальных координат.
// Поэтому serial, OpenMP и MPI стартуют с одинакового поля,
// даже если каждый MPI-процесс заполняет только свой кусок.
// Примерно 25% клеток живые.
inline Cell initial_cell(int i, int j)
{
    unsigned h = (unsigned)i * 73856093u ^ (unsigned)j * 19349663u ^ 42u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return (h % 4) == 0;
}

// Количество живых соседей клетки (i, j). Столбцы "заворачиваются".
inline int count_neighbors(const Grid &g, int i, int j, int n)
{
    int jl = (j == 0) ? n - 1 : j - 1;      // столбец слева
    int jr = (j == n - 1) ? 0 : j + 1;      // столбец справа
    return at(g, i - 1, jl, n) + at(g, i - 1, j, n) + at(g, i - 1, jr, n)
         + at(g, i,     jl, n)                      + at(g, i,     jr, n)
         + at(g, i + 1, jl, n) + at(g, i + 1, j, n) + at(g, i + 1, jr, n);
}

// Правило Конвея: новое состояние клетки.
inline Cell life_rule(Cell alive, int neighbors)
{
    return (neighbors == 3) || (alive && neighbors == 2);
}

// Аргументы командной строки: N STEPS [output_file]
struct Params {
    int n = 1000;           // поле N x N
    int steps = 100;        // число поколений
    std::string out;        // куда сохранить итоговое поле (необязательно)
};

inline Params parse_args(int argc, char **argv)
{
    Params p;
    if (argc > 1) p.n = std::atoi(argv[1]);
    if (argc > 2) p.steps = std::atoi(argv[2]);
    if (argc > 3) p.out = argv[3];
    if (p.n < 3 || p.steps < 0) {
        std::cerr << "Usage: " << argv[0] << " N STEPS [output_file]\n";
        std::exit(1);
    }
    return p;
}

// Сохранить настоящие строки 1..rows в текстовый файл ('0'/'1').
inline void save_grid(const std::string &file, const Grid &g, int rows, int n)
{
    std::ofstream f(file);
    for (int i = 1; i <= rows; i++) {
        for (int j = 0; j < n; j++) f.put('0' + at(g, i, j, n));
        f.put('\n');
    }
}

// Печать результата в одном формате для всех версий (его читает benchmark.py).
inline void print_result(const char *version, const Params &p, int workers,
                         double seconds, long long alive)
{
    std::cout << "version=" << version << " N=" << p.n << " steps=" << p.steps
              << " workers=" << workers << " time=" << seconds
              << " alive=" << alive << std::endl;
}
