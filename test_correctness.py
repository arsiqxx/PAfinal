"""
test_correctness.py - проверка, что параллельные версии дают ТОТ ЖЕ результат,
что и последовательная (клетка в клетку).

1) Тест "глайдер" и "мигалка": проверка самих правил на известных фигурах
   делается через сравнение serial с ручной Python-реализацией.
2) Для нескольких размеров поля и разного числа потоков/процессов
   итоговые поля сравниваются побайтово с serial.

Запуск:  python test_correctness.py
"""
import os
import subprocess
import sys

MSYS = r"C:\msys64\ucrt64\bin"
env = dict(os.environ, PATH=MSYS + os.pathsep + os.environ["PATH"])


HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def run(cmd, threads=1):
    # .exe ищем в папке проекта (полный путь)
    cmd = [os.path.join(HERE, c) if c.endswith(".exe") and c != "mpiexec" else c for c in cmd]
    e = dict(env, OMP_NUM_THREADS=str(threads))
    return subprocess.run(cmd, env=e, capture_output=True, text=True, check=True).stdout


def read(path):
    with open(path) as f:
        return f.read()


def initial_cell(i, j):
    # та же формула, что в life_common.hpp
    m = 0xFFFFFFFF
    h = ((i * 73856093) & m) ^ ((j * 19349663) & m) ^ 42
    h ^= h >> 13
    h = (h * 0x5BD1E995) & m
    h ^= h >> 15
    return 1 if h % 4 == 0 else 0


def python_life(n, steps):
    """Очень простая эталонная реализация на Python (для маленьких полей)."""
    g = [[initial_cell(i, j) for j in range(n)] for i in range(n)]
    for _ in range(steps):
        new = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = sum(g[(i + di) % n][(j + dj) % n]
                        for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)
                new[i][j] = 1 if s == 3 or (g[i][j] and s == 2) else 0
        g = new
    return "".join("".join(map(str, row)) + "\n" for row in g)


failed = 0


def check(name, ok):
    global failed
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failed += 1


# --- 1. serial против независимой Python-реализации ---
run(["life_serial.exe", "40", "30", "ref.txt"])
check("serial == python reference (40x40, 30 steps)", read("ref.txt") == python_life(40, 30))

# --- 2. OpenMP и MPI против serial ---
for n, steps in [(37, 25), (200, 50), (513, 20)]:      # в т.ч. размеры, не делящиеся нацело
    run(["life_serial.exe", str(n), str(steps), "ref.txt"])
    ref = read("ref.txt")
    for t in [1, 2, 3, 4, 8]:
        run(["life_omp.exe", str(n), str(steps), "out.txt"], threads=t)
        check(f"openmp  N={n} steps={steps} threads={t}", read("out.txt") == ref)
    for p in [1, 2, 3, 4, 7]:
        for t in [1, 2]:
            run(["mpiexec", "-n", str(p), "life_mpi.exe", str(n), str(steps), "out.txt"], threads=t)
            check(f"mpi     N={n} steps={steps} procs={p} threads={t}", read("out.txt") == ref)

for f in ("ref.txt", "out.txt"):
    if os.path.exists(f):
        os.remove(f)

print("\nALL TESTS PASSED" if failed == 0 else f"\n{failed} TEST(S) FAILED")
sys.exit(1 if failed else 0)
