//Uses merge sort for each process
//Run this to run the code
//
//mpic++ Problem1.cpp -o p1.x
//mpirun -n 4 ./p1.x
//
//The 4 above is how many slots it uses (change to 1 for simple tests)

#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

//Merge two vectors into one
vector<int> merge(const vector<int>& left, const vector<int>& right) 
{
    vector<int> result;
    int i = 0, j = 0;

    while (i < left.size() && j < right.size()) 
    {
        if (left[i] <= right[j])
            result.push_back(left[i++]);
        else
            result.push_back(right[j++]);
    }

    while (i < left.size()) result.push_back(left[i++]);
    while (j < right.size()) result.push_back(right[j++]);

    return result;
}

//Merge sort
vector<int> mergeSort(const vector<int>& arr) 
{
    //Size 1 is already sorted
    if (arr.size() <= 1)
        return arr;

    //Split into halves
    int mid = arr.size() / 2;
    vector<int> left(arr.begin(), arr.begin() + mid);
    vector<int> right(arr.begin() + mid, arr.end());

    //Recursively sort each half
    left = mergeSort(left);
    right = mergeSort(right);

    //Merge two halves
    return merge(left, right);
}

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 16;
    vector<int> data(N);

    if (rank == 0) 
    {
        data = {34, 7, 23, 32, 5, 62, 78, 1, 88, 14, 9, 45, 3, 99, 56, 21};
        cout << "Original array: ";
        for (int x : data) cout << x << " ";
        cout << endl;
    }

    //Scatterto each process
    int chunk_size = N / size;
    vector<int> local_data(chunk_size);

    MPI_Scatter(data.data(), chunk_size, MPI_INT,
                local_data.data(), chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    //Each process runs merge sort
    local_data = mergeSort(local_data);

    cout << "Rank " << rank << " sorted chunk: ";
    for (int x : local_data) cout << x << " ";
    cout << endl;

    //Binary tree reduction
    int step = 1;
    while (step < size) 
    {
        if (rank % (2 * step) == 0) 
        {
            //Receives and merges
            int partner = rank + step;
            if (partner < size) 
            {
                int incoming_size;
                MPI_Recv(&incoming_size, 1, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                vector<int> incoming(incoming_size);
                MPI_Recv(incoming.data(), incoming_size, MPI_INT, partner, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                local_data = merge(local_data, incoming);
            }
        } 
        else 
        {
            //Sends chunk to partner
            int partner = rank - (rank % (2 * step));
            int send_size = local_data.size();
            MPI_Send(&send_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(local_data.data(), send_size, MPI_INT, partner, 1,
                     MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    //Print final array
    if (rank == 0) {
        cout << "\nFinal sorted array: ";
        for (int x : local_data) cout << x << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
