//Problem 6
//MPI Tests
//
//Run it like this
//
//mpic++ Problem6.cpp -o p6.x
//mpirun -n 1 ./p6.x            The 1 is the processes

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

//This prints a vector
void print_vec(const string &label, int rank, const vector<int> &v)
{
    cout << label << " [rank " << rank << "]:";
    for (int x : v) cout << " " << x;
    cout << endl;
}

int main(int argc, char **argv)
{
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //1. Bcast + Reduce vs Allreduce
    if (0 == rank)
        cout << "\n=== Task 1 ===" << endl;

    int local_val = rank + 1;//each process contributes

    //MPI_Reduce then MPI_Bcast
    int sum_manual = 0;
    MPI_Reduce(&local_val, &sum_manual, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum_manual, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //MPI_Allreduce
    int sum_allreduce = 0;
    MPI_Allreduce(&local_val, &sum_allreduce, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    //Check answers
    int expected_sum = size * (size + 1) / 2;
    if (sum_manual != sum_allreduce || sum_manual != expected_sum)
    {
        //They produce different answers
        cerr << "ERROR: rank " << rank << " manual = " << sum_manual << " allreduce = " << sum_allreduce << " expected = " << expected_sum << endl;
    }
    else
    {
        //They produce the same answers
        cout << "They Match: rank " << rank << ": Reduce+Bcast = " << sum_manual << ", Allreduce = " << sum_allreduce << " expected = " << expected_sum << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);



    //2. Gather + Bcast  vs  Allgather
    if (0 == rank)
        cout << "\n=== Task 2 ===" << endl;

    int my_element = rank * 10;
    vector<int> gathered_manual(size, 0);
    vector<int> gathered_allgather(size, 0);

    //MPI_Gather then MPI_Bcast
    MPI_Gather(&my_element, 1, MPI_INT, gathered_manual.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gathered_manual.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    //MPI_Allgather
    MPI_Allgather(&my_element, 1, MPI_INT, gathered_allgather.data(), 1, MPI_INT, MPI_COMM_WORLD);

    //Check Answers
    bool match2 = (gathered_manual == gathered_allgather);
    //Do they match? VVV
    cout << "rank " << rank << ": Gather + Bcast vs Allgather match = " << (match2 ? "YES" : "NO") << endl;
    if (0 == rank)
    {
        print_vec("  Gather + Bcast result", rank, gathered_manual);
        print_vec("  Allgather result   ", rank, gathered_allgather);
    }

    MPI_Barrier(MPI_COMM_WORLD);



    //3. MPI_Alltoall in MPI_COMM_WORLD
    if (0 == rank)
        cout << "\n=== Task 3 ===" << endl;

    vector<int> send_buf(size), recv_buf(size);

    //q = rank * 100 + q
    for (int q = 0; q < size; ++q)
        send_buf[q] = rank * 100 + q;

    MPI_Alltoall(send_buf.data(), 1, MPI_INT, recv_buf.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Verify: recv_buf[q] should equal q*100 + rank
    bool ok3 = true;
    for (int q = 0; q < size; ++q)
    {
        int expected_from_q = q * 100 + rank;
        //If not correct
        if (recv_buf[q] != expected_from_q)
        {
            cerr << "ERROR: rank " << rank << " recv_buf[" << q << "] = " << recv_buf[q] << " expected = " << expected_from_q << endl;
            ok3 = false;
        }
    }

    //Show results
    cout << "rank " << rank << ": sent";
    for (int x : send_buf) cout << " " << x;
    cout << " | received";
    for (int x : recv_buf) cout << " " << x;
    cout << " | correct = " << (ok3 ? "YES" : "NO") << endl;

    MPI_Finalize();
    cout << "THANK YOU!!!";
    return 0;
}