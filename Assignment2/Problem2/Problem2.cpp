//How to run
//
//mkdir build
//cd build
//cmake -DCMAKE_INSTALL_PREFIX=. ..
//make -j4 install
//./Problem2.cpp


#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) 
{
    Kokkos::initialize(argc, argv);
    {
        //Create 1D View of doubles
        Kokkos::View<double*> my_view("Problem2", 10);

        //Print label
        std::cout << "View label: " << my_view.label() << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
