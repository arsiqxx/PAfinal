
import csv
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
env = dict(os.environ, PATH=r"C:\msys64\ucrt64\bin" + os.pathsep + os.environ["PATH"])

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
REPEATS = 3
CPUS = os.cpu_count() or 4
WORKERS = [p for p in [1, 2, 4, 8, 16] if p <= CPUS]


def run_once(cmd, threads):
    e = dict(env, OMP_NUM_THREADS=str(threads))
    out = subprocess.run(cmd, env=e, capture_output=True, text=True, check=True).stdout
    fields = dict(kv.split("=") for kv in out.split())
    return float(fields["time"])


def measure(kind, n, p):
    exe = lambda name: os.path.join(HERE, name)
    if kind == "serial":
        cmd, t = [exe("life_serial.exe"), str(n), str(STEPS)], 1
    elif kind == "openmp":
        cmd, t = [exe("life_omp.exe"), str(n), str(STEPS)], p
    elif kind == "mpi":
        cmd, t = ["mpiexec", "-n", str(p), exe("life_mpi.exe"), str(n), str(STEPS)], 1
    elif kind == "hybrid":   # 2 MPI-процесса, в каждом p/2 потоков
        cmd, t = ["mpiexec", "-n", "2", exe("life_mpi.exe"), str(n), str(STEPS)], max(1, p // 2)
    best = min(run_once(cmd, t) for _ in range(REPEATS))
    print(f"{kind:7s} N={n:5d} p={p:2d}  time={best:.4f} s")
    return best


rows = []
KINDS = ["openmp", "mpi", "hybrid"]

t_serial = measure("serial", N, 1)
rows.append(["strong", "serial", N, 1, t_serial, 1.0, 1.0])
strong = {k: [] for k in KINDS}
for k in KINDS:
    for p in WORKERS:
        if k == "hybrid" and p < 2:
            continue
        t = measure(k, N, p)
        s = t_serial / t
        strong[k].append((p, t, s, s / p))
        rows.append(["strong", k, N, p, t, s, s / p])

base = N // 2
weak = {k: [] for k in ["openmp", "mpi"]}
for k in weak:
    for p in WORKERS:
        n = int(base * math.sqrt(p))
        t = measure(k, n, p)
        weak[k].append((p, t))
        rows.append(["weak", k, n, p, t, "", ""])

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment", "version", "N", "workers", "time_s", "speedup", "efficiency"])
    w.writerows(rows)

labels = {"openmp": "OpenMP (threads)", "mpi": "MPI (processes)", "hybrid": "MPI+OpenMP (2 procs x threads)"}


def plot(index, ylabel, title, fname, ideal):
    plt.figure(figsize=(6, 4))
    for k in KINDS:
        xs = [r[0] for r in strong[k]]
        plt.plot(xs, [r[index] for r in strong[k]], "o-", label=labels[k])
    if ideal == "linear":
        plt.plot(WORKERS, WORKERS, "k--", label="ideal")
    else:
        plt.axhline(1.0, color="k", linestyle="--", label="ideal")
    plt.xlabel("threads / processes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(WORKERS)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()


plot(1, "time, s", f"Execution time (N={N}, {STEPS} steps)", "time.png", None)
plot(2, "speedup", f"Speedup (N={N}, {STEPS} steps)", "speedup.png", "linear")
plot(3, "efficiency", f"Efficiency (N={N}, {STEPS} steps)", "efficiency.png", "one")

plt.figure(figsize=(6, 4))
for k in weak:
    plt.plot([r[0] for r in weak[k]], [r[1] for r in weak[k]], "o-", label=labels[k])
plt.xlabel("threads / processes")
plt.ylabel("time, s (ideal = flat line)")
plt.title(f"Weak scaling ({base}x{base} cells per worker)")
plt.xticks(WORKERS)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("weak_scaling.png", dpi=120)

print("\nSaved: results.csv, time.png, speedup.png, efficiency.png, weak_scaling.png")
