//Problem 1
//Message Passing Circle
//
//Run it like this
//
//mpic++ Problem1.cpp -o p1.x
//mpirun -n 1 ./p1.x                The 1 is the number of slots
//
//You could change N without touching the code, using "mpic++ problem1.cpp -DRING_TRIPS=5 -o p1.x"

#include <mpi.h>
#include <iostream>

using namespace std;

#ifndef RING_TRIPS
#define RING_TRIPS 3 //This is N, I got it as 3 rn
#endif

int main(int argc, char **argv)
{
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //N is the number of times you go in a circle
    const int N = RING_TRIPS;
    int value = 0;

    if (0 == rank)
        cout << "==========\n";
        cout << "Running ring with N = " << N << " trips and " << size << " tasks" << endl;

    for (int trip = 0; trip < N; ++trip)
    {
        if (0 == rank)
        {
            //Process 1 gets sent the current value
            cout << "task 0 now sending number: " << value << " to task 1" << endl;
            MPI_Send(&value, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);

            //This recieves the value after it goes through the loop
            int received;
            MPI_Recv(&received, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            value = received + 1;
            cout << "task 0 received number: " << received << " from task " << (size - 1) << " now incrementing it to " << value << endl;
        }
        else
        {
            //Recieve from previous process
            int received;
            MPI_Recv(&received, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cout << "task " << rank << " received number: " << received << " from task " << (rank - 1) << endl;

            //Send forward
            cout << "sending number: " << received << " to task " << (rank + 1) % size << endl;
            MPI_Send(&received, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
        }
    }

    if (0 == rank)
        cout << "passing complete; number is " << value << endl;

    MPI_Finalize();
    cout << "\n\n\nTHANK YOU!!!\n";
    return 0;
}