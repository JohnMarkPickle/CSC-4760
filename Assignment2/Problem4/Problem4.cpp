#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) 
{
    Kokkos::initialize(argc, argv);
    {
        int n = 20;

        Kokkos::View<double****> A("A", 5, 7, 12, n);

        std::cout << "View label: " << A.label() << std::endl;
        std::cout << "Dimensions: " 
        << A.extent(0) << "x" 
        << A.extent(1) << "x" 
        << A.extent(2) << "x" 
        << A.extent(3) << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
