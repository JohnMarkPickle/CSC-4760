//Problem 3
//Vector-Vector
//
//Run it like this
//
//mpic++ Problem3.cpp -o p3.x
//mpirun -n 1 ./p3.x 1024        The 1 and 1024 can be changed to desired values (1 is slots)

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

//Shows how many elements process p owns
void partitionSize(int P, int N, int p, int &entries)
{
    int L = N / P;
    int R = N % P;
    entries = (p < R) ? (L + 1) : L;
}

//Process p's index displacement
void partitionDispl(int P, int N, int p, int &displ)
{
    int L = N / P;
    int R = N % P;
    //processes have (L+1) elements
    if (p < R)
        displ = p * (L + 1);
    else
        displ = R * (L + 1) + (p - R) * L;
}

int main(int argc, char **argv)
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Finds N
    int N = 1024;
    if (argc >= 2)
        N = atoi(argv[1]);

    if (0 == rank)
        cout << "N = " << N << ", P = " << size << endl;

    //This builds the vector
    vector<double> A_global, B_global;
    if (0 == rank)
    {
        A_global.resize(N);
        B_global.resize(N);
        for (int i = 0; i < N; ++i)
        {
            A_global[i] = static_cast<double>(i + 1);
            B_global[i] = 1.0;
        }
        double expected = static_cast<double>(N) * (N + 1) / 2.0;
        //This is what is expected VVV
        cout << "Dot product = " << expected << endl;
    }

    //Process 0 sends stuff to the other processes
    int my_count, my_displ;
    partitionSize(size, N, rank, my_count);
    partitionDispl(size, N, rank, my_displ);
    vector<double> A_local(my_count), B_local(my_count);

    if (0 == rank)
    {
        //Copies itself
        for (int i = 0; i < my_count; ++i)
        {
            A_local[i] = A_global[my_displ + i];
            B_local[i] = B_global[my_displ + i];
        }
        //Sends it to other processes
        for (int p = 1; p < size; ++p)
        {
            int cnt, dsp;
            partitionSize(size, N, p, cnt);
            partitionDispl(size, N, p, dsp);
            MPI_Send(&A_global[dsp], cnt, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            MPI_Send(&B_global[dsp], cnt, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(A_local.data(), my_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_local.data(), my_count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    //Multiply and partial sum
    double t_start = MPI_Wtime();
    double partial_sum = 0.0;
    for (int i = 0; i < my_count; ++i)
        partial_sum += A_local[i] * B_local[i];

    //Processes get their data in a tree-like format
    for (int stride = 1; stride < size; stride *= 2)
    {
        if (rank % (2 * stride) == 0)
        {
            //This is a receiver 
            int partner = rank + stride;
            if (partner < size)
            {
                double incoming;
                MPI_Recv(&incoming, 1, MPI_DOUBLE, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                partial_sum += incoming;
            }
        }
        else if (rank % (2 * stride) == stride)
        {
            //This is a sender
            MPI_Send(&partial_sum, 1, MPI_DOUBLE, rank - stride, 2, MPI_COMM_WORLD);
            break;//After done sending, the process is done
        }
    }

    double t_end = MPI_Wtime();

    //Process 0 sends the results
    if (0 == rank)
    {
        double expected = static_cast<double>(N) * (N + 1) / 2.0;
        cout << "Actual dot product = " << partial_sum << endl;
        cout << "Error = " << fabs(partial_sum - expected) << endl;
        cout << "Time  = " << (t_end - t_start) << " seconds" << endl;
    }
    MPI_Finalize();
    cout << "\n\n\nTHANK YOU!!!\n";
    return 0;
}