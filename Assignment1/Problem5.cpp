//Problem 5
//MPI Comm Split
//
//Run it like this
//
//mpic++ Problem5.cpp -o p5.x
//mpirun -n 1 ./p5.x            The world size is P * Q = 1, but replace the 1 with the size you want. P * Q must equal it though

#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //P * Q = Size
    int P = 0, Q = 0;
    if (0 == rank)
    {
        do
        {
            cout << "Enter dimensions P and Q (P*Q = " << size << "): ";
            cin >> P >> Q;
        } 
        while (P * Q != size || P < 1 || Q < 1);
    }

    int share[2] = {P, Q};
    MPI_Bcast(share, 2, MPI_INT, 0, MPI_COMM_WORLD);
    P = share[0];
    Q = share[1];

    MPI_Comm rowcomm;
    //Split is rank / Q
    MPI_Comm_split(MPI_COMM_WORLD, rank / Q, rank, &rowcomm);

    int row_rank, row_size;
    MPI_Comm_rank(rowcomm, &row_rank);
    MPI_Comm_size(rowcomm, &row_size);

    MPI_Comm colcomm;
    //Split is rank % Q
    MPI_Comm_split(MPI_COMM_WORLD, rank % Q, rank, &colcomm);

    int col_rank, col_size;
    MPI_Comm_rank(colcomm, &col_rank);
    MPI_Comm_size(colcomm, &col_size);

    //Show grid location
    cout << "World rank " << rank << ": grid position (" << col_rank << ", " << row_rank << ")" << " | rowcomm size=" << row_size << " | colcomm size=" << col_size << endl;

    int row_root_world_rank = rank - row_rank;
    MPI_Bcast(&row_root_world_rank, 1, MPI_INT, 0, rowcomm);

    //Rank 0 sends it's rank to others
    int col_root_world_rank = rank - col_rank * Q;
    MPI_Bcast(&col_root_world_rank, 1, MPI_INT, 0, colcomm);

    cout << "World rank " << rank << ": row root is world rank " << row_root_world_rank << ", col root is world rank " << col_root_world_rank << endl;

    MPI_Comm_free(&rowcomm);
    MPI_Comm_free(&colcomm);

    MPI_Finalize();
    cout << "\n\n\nTHANK YOU!!!\n";
    return 0;
}