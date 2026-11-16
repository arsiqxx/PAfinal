@echo off
rem Build all 3 versions with MSYS2 UCRT64 g++ and Microsoft MPI (no WSL needed)
set PATH=C:\msys64\ucrt64\bin;%PATH%
set FLAGS=-O2 -std=c++17 -Wall

g++ %FLAGS% life_serial.cpp -o life_serial.exe || exit /b 1
g++ %FLAGS% -fopenmp life_omp.cpp -o life_omp.exe || exit /b 1
g++ %FLAGS% -fopenmp life_mpi.cpp -o life_mpi.exe -lmsmpi || exit /b 1

echo Build OK: life_serial.exe life_omp.exe life_mpi.exe

