//problem1.cpp
//
//MPI_Comm_split x MPI_COMM_WORLD
//
//How to run
//
//mpicxx -Wall -o p1 problem1.cpp
//mpirun -np 6  ./p1 2 3
//
//That's a 2 x 3 grid with 6 processes

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //get P and Q from command line
    if (argc != 3) 
    {
        if (world_rank == 0)
            fprintf(stderr, "How to use: mpirun -np <P*Q> %s <P> <Q>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int P = atoi(argv[1]);
    int Q = atoi(argv[2]);

    if (P < 1 || Q < 1) 
    {
        if (world_rank == 0)
            fprintf(stderr, "Error: P and Q must be >= 1.\n");
        MPI_Finalize();
        return 1;
    }

    if (P * Q != world_size) 
    {
        if (world_rank == 0)
            fprintf(stderr,
                    "Error: P*Q (%d*%d = %d) must equal MPI world size (%d).\n",
                    P, Q, P * Q, world_size);
        MPI_Finalize();
        return 1;
    }

    //row communicators 
    int row_color = world_rank / Q;
    int row_key = world_rank % Q;

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, row_key, &row_comm);

    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    int row_sum = 0;
    MPI_Reduce(&world_rank, &row_sum, 1, MPI_INT, MPI_SUM, 0, row_comm);

    if (row_rank == 0) 
    {
        printf("[Split 1 | Row %d | world rank %d] "
               "Sum of world ranks in this row = %d\n",
               row_color, world_rank, row_sum);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) 
    {
        printf("--\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //column communicators
    int col_color = world_rank % Q;
    int col_key = world_rank / Q;

    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, col_key, &col_comm);

    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    int bcast_value = world_rank;
    MPI_Bcast(&bcast_value, 1, MPI_INT, 0, col_comm);

    //Print everything
    printf("[Split 2 | Col %d | world rank %d | col rank %d] "
           "bcast value = %d (from world rank %d)\n",
           col_color, world_rank, col_rank, bcast_value,
           col_color);
    fflush(stdout);

    //Cleanup
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}
