#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

//Merges two halves into one sorted vector
vector<int> merge(const vector<int>& left, const vector<int>& right) 
{
    vector<int> result;
    int i = 0, j = 0;

    //Compare elements from each half
    while (i < left.size() && j < right.size()) 
    {
        if (left[i] <= right[j])
            result.push_back(left[i++]);
        else
            result.push_back(right[j++]);
    }

    //Add any elements left
    while (i < left.size()) result.push_back(left[i++]);
    while (j < right.size()) result.push_back(right[j++]);

    return result;
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //The full array for process 0
    const int N = 16;
    vector<int> data(N);

    if (rank == 0) 
    {
        //Fill with unsorted values
        data = {34, 7, 23, 32, 5, 62, 78, 1, 88, 14, 9, 45, 3, 99, 56, 21};
        cout << "Original array: ";
        for (int x : data) cout << x << " ";
        cout << endl;
    }

    //Scatter array across all processes
    int chunk_size = N / size;
    vector<int> local_data(chunk_size);

    //MPI_Scatter
    MPI_Scatter(data.data(), chunk_size, MPI_INT,
                local_data.data(), chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    //Each process sorts
    sort(local_data.begin(), local_data.end());

    cout << "Rank " << rank << " sorted chunk: ";
    for (int x : local_data) cout << x << " ";
    cout << endl;

    //Binary tree reduction
    int step = 1;
    while (step < size) 
    {
        if (rank % (2 * step) == 0) 
        {
            //This process receives from rank + step
            int partner = rank + step;
            if (partner < size) 
            {
                int incoming_size;
                MPI_Recv(&incoming_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                vector<int> incoming(incoming_size);
                MPI_Recv(incoming.data(), incoming_size, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                local_data = merge(local_data, incoming);
            }
        } 
        else 
        {
            //This process sends to rank - step
            int partner = rank - (rank % (2 * step));
            int send_size = local_data.size();
            MPI_Send(&send_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(local_data.data(), send_size, MPI_INT, partner, 1, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    //Prints final array
    if (rank == 0) 
    {
        cout << "\nFinal sorted array: ";
        for (int x : local_data) cout << x << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}