//problem2.cpp
//
//How to run
//
//mpicxx -Wall -o p2 problem2.cpp
//mpirun -np 8 ./p2 4 2 15
//
//mpirun -np <P*Q> ./p2 <P> <Q> <M>

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

static int wrap_size(int M, int Q, int q)
{
    return lin_size(M, Q, q);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //parse arguments
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
        if (world_rank == 0)
            fprintf(stderr, "Error: P, Q, M must all be >= 1.\n");
        MPI_Finalize();
        return 1;
    }
    if (P * Q != world_size)
    {
        if (world_rank == 0)
            fprintf(stderr, "Error: P*Q (%d*%d=%d) != world size (%d).\n",
                P, Q, P * Q, world_size);
        MPI_Finalize();
        return 1;
    }
    if (M < P)
    {
        if (world_rank == 0)
            fprintf(stderr, "Error: M (%d) must be >= P (%d).\n", M, P);
        MPI_Finalize();
        return 1;
    }

    //(row, col) from world rank
    int my_row = world_rank / Q;
    int my_col = world_rank % Q;

    //Row communicator
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);

    //Column communicator
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    int local_n = lin_size(M, P, my_row);

    double *x_local = (double *)calloc(local_n, sizeof(double));
    if (!x_local)
    {
        fprintf(stderr, "rank %d: malloc failed\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int local_y_n = wrap_size(M, Q, my_col);

    double *y_local = (double *)calloc(local_y_n, sizeof(double));
    if (!y_local)
    {
        fprintf(stderr, "rank %d: malloc failed\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double *x_global = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    if (my_row == 0 && my_col == 0)
    {
        x_global = (double *)malloc(M * sizeof(double));
        sendcounts = (int *)malloc(P * sizeof(int));
        displs = (int *)malloc(P * sizeof(int));
        if (!x_global || !sendcounts || !displs)
        {
            fprintf(stderr, "rank 0: malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < M; i++)
            x_global[i] = (double)i;

        for (int r = 0; r < P; r++)
        {
            sendcounts[r] = lin_size(M, P, r);
            displs[r]     = lin_start(M, P, r);
        }

        printf("Process (0,0): full x = [");
        for (int i = 0; i < M; i++)
            printf("%.0f%s", x_global[i], i < M - 1 ? "," : "]\n");
        fflush(stdout);
    }

    if (my_col == 0)
    {
        MPI_Scatterv(x_global, sendcounts, displs, MPI_DOUBLE, x_local, local_n, MPI_DOUBLE, 0, col_comm);
        printf("Process (%d,%d) received x chunk: [", my_row, my_col);
        for (int i = 0; i < local_n; i++)
            printf("%.0f%s", x_local[i], i < local_n - 1 ? "," : "]\n");
        fflush(stdout);
    }

    free(x_global);
    free(sendcounts);
    free(displs);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(x_local, local_n, MPI_DOUBLE, 0, row_comm);

    printf("Process (%d,%d) after bcast: x = [", my_row, my_col);
    for (int i = 0; i < local_n; i++)
        printf("%.0f%s", x_local[i], i < local_n - 1 ? "," : "]\n");
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);

    int row_start = lin_start(M, P, my_row);

    //Count elements this process sends
    int *s_counts = (int *)calloc(Q, sizeof(int));
    int *r_counts = (int *)calloc(Q, sizeof(int));
    int *s_displs = (int *)calloc(Q, sizeof(int));
    int *r_displs = (int *)calloc(Q, sizeof(int));
    if (!s_counts || !r_counts || !s_displs || !r_displs)
        MPI_Abort(MPI_COMM_WORLD, 1);

    for (int i = 0; i < local_n; i++)
    {
        int J = row_start + i;
        int dest = J % Q;
        s_counts[dest]++;
    }

    //Exchange counts
    MPI_Alltoall(s_counts, 1, MPI_INT, r_counts, 1, MPI_INT, row_comm);

    //Build displacements
    s_displs[0] = r_displs[0] = 0;
    for (int q = 1; q < Q; q++)
    {
        s_displs[q] = s_displs[q - 1] + s_counts[q - 1];
        r_displs[q] = r_displs[q - 1] + r_counts[q - 1];
    }
    int total_send = s_displs[Q - 1] + s_counts[Q - 1];
    int total_recv = r_displs[Q - 1] + r_counts[Q - 1];

    double *send_buf = (double *)malloc(total_send * sizeof(double));
    double *recv_buf = (double *)malloc(total_recv * sizeof(double));
    if (!send_buf || !recv_buf)
        MPI_Abort(MPI_COMM_WORLD, 1);

    int *cursor = (int *)calloc(Q, sizeof(int));
    if (!cursor) MPI_Abort(MPI_COMM_WORLD, 1);

    for (int i = 0; i < local_n; i++)
    {
        int J = row_start + i;
        int dest = J % Q;
        send_buf[s_displs[dest] + cursor[dest]] = x_local[i];
        cursor[dest]++;
    }
    free(cursor);

    MPI_Alltoallv(send_buf, s_counts, s_displs, MPI_DOUBLE, recv_buf, r_counts, r_displs, MPI_DOUBLE, row_comm);

    free(send_buf);

    for (int q_src = 0; q_src < Q; q_src++)
    {
        int count = r_counts[q_src];
        int base = r_displs[q_src];

        int rem = (int)((row_start % Q + Q) % Q);
        int offset = (my_col - rem + Q) % Q;
        int first_J = row_start + offset;

        for (int k = 0; k < count; k++)
        {
            int J = first_J + k * Q;
            int j = J / Q;
            y_local[j] = recv_buf[base + k];
        }
    }

    free(recv_buf);
    free(s_counts);
    free(r_counts);
    free(s_displs);
    free(r_displs);

    printf("Process (%d,%d): y (wrap-mapped, q=%d) = [", my_row, my_col, my_col);
    for (int j = 0; j < local_y_n; j++)
        printf("%.0f%s", y_local[j], j < local_y_n - 1 ? "," : "]\n");
    fflush(stdout);

    int ok = 1;
    for (int j = 0; j < local_y_n; j++)
    {
        double expected = (double)(j * Q + my_col);
        if (y_local[j] != expected) { ok = 0; break; }
    }
    int global_ok;
    MPI_Allreduce(&ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (world_rank == 0)
        printf("\ny == x (wrap-mapped) on all processes: %s\n",
               global_ok ? "Yes!" : "No...");

    //Cleanup
    free(x_local);
    free(y_local);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}
