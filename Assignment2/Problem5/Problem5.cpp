#include <Kokkos_Core.hpp>
#include <iostream>
#include <climits>

int main(int argc, char* argv[]) 
{
    Kokkos::initialize(argc, argv);
    {
        const int N = 10;

        //Create a 1D View
        Kokkos::View<int*> A("A", N);

        //Populate with A(i) = i * 10
        Kokkos::parallel_for
        (
            "FillA",
            N,
            KOKKOS_LAMBDA(const int i) 
            {
                A(i) = i * 10;
            }
        );

        //Reduce to find the max
        int max_val = INT_MIN;
        Kokkos::parallel_reduce
        (
            "MaxReduce",
            N,
            KOKKOS_LAMBDA(const int i, int& local_max) 
            {
                if (A(i) > local_max) 
                {
                    local_max = A(i);
                }
            },
            Kokkos::Max<int>(max_val)
        );
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
        std::cout << "Array A: ";
        for(int i = 0; i < N; i++)
            {
                std::cout << A_host(i) << " ";
            }
        std::cout << "\n";
        std::cout << "Maximum of A = " << max_val << "\n";
    }
    Kokkos::finalize();
    return 0;
}
