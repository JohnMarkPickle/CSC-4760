//problem6.cpp
//
//How to run
//
//mpicxx -Wall -o p6 problem6.cpp
//mpirun -np 8 ./p6 4 2 15
//
//mpirun -np <P*Q> ./p6 <P> <Q> <M>

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int lin_size(int M, int P, int rank)
{
    return M / P + (rank < M % P ? 1 : 0);
}

static int lin_start(int M, int P, int rank)
{
    int base = M / P, rem = M % P;
    return rank * base + (rank < rem ? rank : rem);
}

static double expected_dot(int M)
{
    double m = (double)M;
    return 2.0 * m * (m - 1.0) * (2.0 * m - 1.0) / 6.0 + m * (m - 1.0) / 2.0;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 4) 
    {
        if (world_rank == 0)
            fprintf(stderr, "How to use: mpirun -np <P*Q> %s <P> <Q> <M>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int P = atoi(argv[1]);
    int Q = atoi(argv[2]);
    int M = atoi(argv[3]);

    if (P < 1 || Q < 1 || M < 1) 
    {
        if (world_rank == 0) fprintf(stderr, "Error: P, Q, M must be >= 1.\n");
        MPI_Finalize(); return 1;
    }
    if (P * Q != world_size) 
    {
        if (world_rank == 0)
            fprintf(stderr, "Error: P*Q = %d * %d = %d but world size = %d.\n",
                    P, Q, P*Q, world_size);
        MPI_Finalize(); return 1;
    }
    if (M < P || M < Q) 
    {
        if (world_rank == 0)
            fprintf(stderr, "Error: M = %d must be >= P = %d and >= Q = %d.\n", M, P, Q);
        MPI_Finalize(); return 1;
    }

    int my_row = world_rank / Q;
    int my_col = world_rank % Q;

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);

    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);

    int x_row_n  = lin_size (M, P, my_row);
    int row_start = lin_start(M, P, my_row);
    int z_col_n  = lin_size (M, Q, my_col);
    int col_start = lin_start(M, Q, my_col);

    double *x_local = (double *)calloc(x_row_n, sizeof(double));
    if (!x_local) { fprintf(stderr,"rank %d: OOM x_local\n",world_rank); MPI_Abort(MPI_COMM_WORLD,1); }

    {
        double *x_global = NULL;
        int *scounts = NULL;
        int *sdispls = NULL;

        if (world_rank == 0) 
        {
            x_global = (double *)malloc(M * sizeof(double));
            scounts = (int *)malloc(P * sizeof(int));
            sdispls = (int *)malloc(P * sizeof(int));
            if (!x_global || !scounts || !sdispls)
                MPI_Abort(MPI_COMM_WORLD, 1);

            for (int i = 0; i < M; i++)
                x_global[i] = (double)i;

            for (int r = 0; r < P; r++) 
            {
                scounts[r] = lin_size (M, P, r);
                sdispls[r] = lin_start(M, P, r);
            }
            printf("x = [");
            for (int i = 0; i < M; i++)
                printf("%.0f%s", x_global[i], i < M-1 ? "," : "]\n");
            fflush(stdout);
        }

        if (my_col == 0)
            MPI_Scatterv(x_global, scounts, sdispls, MPI_DOUBLE, x_local, x_row_n, MPI_DOUBLE, 0, col_comm);

        free(x_global); free(scounts); free(sdispls);
    }

    MPI_Bcast(x_local, x_row_n, MPI_DOUBLE, 0, row_comm);

    double *z_local = (double *)calloc(z_col_n, sizeof(double));
    if (!z_local) {fprintf(stderr,"rank %d: OOM z_local\n",world_rank); MPI_Abort(MPI_COMM_WORLD,1);}

    {
        double *z_global = NULL;
        int *scounts = NULL;
        int *sdispls = NULL;

        if (world_rank == 0) 
        {
            z_global = (double *)malloc(M * sizeof(double));
            scounts = (int *)malloc(Q * sizeof(int));
            sdispls = (int *)malloc(Q * sizeof(int));
            if (!z_global || !scounts || !sdispls)
                MPI_Abort(MPI_COMM_WORLD, 1);

            for (int i = 0; i < M; i++)
                z_global[i] = 2.0 * i + 1.0;

            for (int c = 0; c < Q; c++) 
            {
                scounts[c] = lin_size (M, Q, c);
                sdispls[c] = lin_start(M, Q, c);
            }
            printf("z = [");
            for (int i = 0; i < M; i++)
                printf("%.0f%s", z_global[i], i < M-1 ? "," : "]\n");
            fflush(stdout);
        }

        if (my_row == 0)
            MPI_Scatterv(z_global, scounts, sdispls, MPI_DOUBLE, z_local, z_col_n, MPI_DOUBLE, 0, row_comm);

        free(z_global); free(scounts); free(sdispls);
    }

    MPI_Bcast(z_local, z_col_n, MPI_DOUBLE, 0, col_comm);

    MPI_Barrier(MPI_COMM_WORLD);

    int row_end = row_start + x_row_n;
    int col_end = col_start + z_col_n;
    int ovl_start = (row_start > col_start) ? row_start : col_start;
    int ovl_end = (row_end < col_end) ? row_end : col_end;
    int ovl_n = (ovl_end > ovl_start) ? ovl_end - ovl_start : 0;

    double partial = 0.0;
    for (int k = 0; k < ovl_n; k++) 
    {
        int x_off = ovl_start - row_start;
        int z_off = ovl_start - col_start;
        partial += x_local[x_off + k] * z_local[z_off + k];
    }

    double dot_product = 0.0;
    MPI_Allreduce(&partial, &dot_product, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int r = 0; r < world_size; r++) 
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == r) 
        {
            printf("[(%d,%d)] x chunk [%d..%d], z chunk [%d..%d], overlap [%d..%d], partial=%.2f\n", my_row, my_col, row_start, row_end - 1, col_start, col_end - 1, ovl_start, ovl_end - 1, partial);
            fflush(stdout);
        }
    }

    if (world_rank == 0) 
    {
        double exp_val = expected_dot(M);
        printf("\n--- Result ---\n");
        printf("Dot product = %.6f\n", dot_product);
        printf("Expected = %.6f\n", exp_val);
        printf("Correct? %s\n", (dot_product == exp_val) ? "Yes!" : "No...");
        printf("Error = %.6e\n", dot_product - exp_val);
        fflush(stdout);
    }

    //cleanup
    free(x_local);
    free(z_local);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
