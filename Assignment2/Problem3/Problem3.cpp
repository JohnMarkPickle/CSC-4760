#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) 
{
    Kokkos::initialize(argc, argv);
    {
        const int n = 5;
        const int m = 6;

        //Create 2D View
        Kokkos::View<int**> A("A", n, m);

        //A(i,j) = 1000 * i * j
        Kokkos::parallel_for
        (
            "FillA",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n,m}),
            KOKKOS_LAMBDA(const int i, const int j) 
            {
                A(i,j) = 1000 * i * j;
            }
        );

        //Copy for printing
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);

        //Print
        std::cout << "A(i,j) = 1000 * i * j\n";
        for(int i = 0; i < n; i++) 
        {
            for(int j = 0; j < m; j++) 
            {
                std::cout << A_host(i,j) << " ";
            }
            std::cout << "\n";
        }
    }
    Kokkos::finalize();
    return 0;
}
