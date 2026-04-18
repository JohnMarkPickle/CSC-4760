#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) 
{
    Kokkos::initialize(argc, argv);
    {
        const int N = 3;
        const int M = 3;

        //Matrix A is 3x3
        Kokkos::View<int**> A("A", N, M);

        //Vector B is size 3
        Kokkos::View<int*> B("B", M);

        //Test case A
        auto A_host = Kokkos::create_mirror_view(A);
        A_host(0,0)=130; A_host(0,1)=147; A_host(0,2)=115;
        A_host(1,0)=224; A_host(1,1)=158; A_host(1,2)=187;
        A_host(2,0)=54;  A_host(2,1)=158; A_host(2,2)=120;
        Kokkos::deep_copy(A, A_host);

        //Test case B
        auto B_host = Kokkos::create_mirror_view(B);
        B_host(0)=221; B_host(1)=12; B_host(2)=157;
        Kokkos::deep_copy(B, B_host);

        //Loop that adds B(j) to A(i,j)
        Kokkos::parallel_for
        (
            "AddVectorToRows",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {N,M}),
            KOKKOS_LAMBDA(int i, int j) 
            {
                A(i,j) += B(j);
            }
        );

        auto A_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);

        //Print
        std::cout << "Answer matrix:\n";
        for(int i = 0; i < N; i++) 
        {
            for(int j = 0; j < M; j++) 
            {
                std::cout << A_result(i,j) << " ";
            }
            std::cout << "\n";
        }
    }
    Kokkos::finalize();
    return 0;
}
