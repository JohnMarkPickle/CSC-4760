//MADE USING COPILOT
/*Prompt
Write a parallel vector-vector element by element multiplication program followed by
a tree reduction to compute the dot-product of two vectors of length N with P processes using MPI. For
instance, show with N = 1024, 2048, 4096 and P = 1, 2, 4.
Use only point-to-point operations. Code your own tree reduction using sends and receives.
Divide the data as equally as you can between your processes. We will discuss how to compute these
partitions in class early next week (week of February 16).
Use MPI Wtime() to measure the cost of the operation as you vary P and N. We will explain how to use
this operation in lecture.
Use easy-to-evaluate data for testing in the vector elements for correctness testing (but not all zeroes).
*/
//
//use these to run
//mpicxx Problem4.cpp -o p4.x
//mpirun -np 4 ./p4.x 1024

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " N\n";
        MPI_Finalize();
        return 1;
    }

    const int N = std::atoi(argv[1]);

    // Compute local partition [start, end)
    int base = N / size;
    int rem  = N % size;

    int local_n = base + (rank < rem ? 1 : 0);
    int start   = rank * base + std::min(rank, rem);
    int end     = start + local_n;

    // Local vectors
    std::vector<double> a(local_n), b(local_n), c(local_n);

    // Easy-to-check data: a[i] = 1, b[i] = global_index + 1
    for (int i = 0; i < local_n; ++i) {
        int global_idx = start + i;
        a[i] = 1.0;
        b[i] = static_cast<double>(global_idx + 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Elementwise multiplication: c = a * b
    for (int i = 0; i < local_n; ++i) {
        c[i] = a[i] * b[i];
    }

    // Local dot product
    double local_dot = 0.0;
    for (int i = 0; i < local_n; ++i) {
        local_dot += c[i];
    }

    // Manual tree reduction using point-to-point
    double result = local_dot;
    int step = 1;
    MPI_Status status;

    while (step < size) {
        if (rank % (2 * step) == 0) {
            int src = rank + step;
            if (src < size) {
                double recv_val;
                MPI_Recv(&recv_val, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &status);
                result += recv_val;
            }
        } else {
            int dest = rank - step;
            MPI_Send(&result, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            break; // Done participating
        }
        step *= 2;
    }

    double t1 = MPI_Wtime();

    if (rank == 0) {
        // Analytical result: sum_{i=1}^N i = N(N+1)/2
        double exact = 0.5 * N * (N + 1);
        std::cout << "N = " << N << ", P = " << size << "\n";
        std::cout << "Dot product (parallel) = " << result << "\n";
        std::cout << "Dot product (exact)    = " << exact << "\n";
        std::cout << "Absolute error         = " << std::fabs(result - exact) << "\n";
        std::cout << "Elapsed time (s)       = " << (t1 - t0) << "\n\n";
    }

    MPI_Finalize();
    return 0;
}