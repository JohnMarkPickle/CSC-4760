//MADE USING COPILOT (Finally a use for the button they put on my keyboard...)
/*Prompt
Write an MPI program that passes a message of one integer around in a logical ring of processes with MPI COMM WORLD.
The integer should start at 0 in process 0 and be incremented each time it passes around the ring,
and you should be able to have the message go around the ring N times, where N is specifed at compile time.
*/
//
//use these to run
//mpic++ Problem2.cpp -o p2.x
//mpirun -n 4 ./p2.x

#include <iostream>
#include <mpi.h>

#define NLOOPS 5   // Number of times the message goes around the ring

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = 0;
    MPI_Status status;

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    for (int loop = 0; loop < NLOOPS; loop++) {

        if (rank == 0) {
            // Rank 0 starts each loop
            if (loop == 0)
                value = 0;

            MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
            MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);

            value++;  // increment after receiving
            std::cout << "Loop " << loop << " complete at rank 0, value = " << value << std::endl;

        } else {
            // All other ranks
            MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
            value++;
            MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}