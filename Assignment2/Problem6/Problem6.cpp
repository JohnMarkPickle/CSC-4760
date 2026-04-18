#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) 
{
    Kokkos::initialize(argc, argv);
    {
        const int N = 5000;
        const int M = 5000;

        Kokkos::View<double**> A("A", N, M);
        Kokkos::View<double*> rowSums_parallel("rowSums_parallel", N);
        Kokkos::View<double*> rowSums_serial("rowSums_serial", N);

        //Fill with A(i,j) = i + j
        Kokkos::parallel_for
        (
            "FillA",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {N,M}),
            KOKKOS_LAMBDA(int i, int j) 
            {
                A(i,j) = i + j;
            }
        );

        //Parallel sum
        Kokkos::Timer timer_parallel;

        Kokkos::parallel_for
        (
            "ParallelRowSum",
            N,
            KOKKOS_LAMBDA(int i) 
            {
                double sum = 0.0;
                for(int j = 0; j < M; j++) 
                {
                    sum += A(i,j);
                }
                rowSums_parallel(i) = sum;
            }
        );

        Kokkos::fence();
        double parallel_time = timer_parallel.seconds();

        //Serial sum
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
        auto rowSums_serial_host = Kokkos::create_mirror_view(rowSums_serial);

        Kokkos::Timer timer_serial;

        for(int i = 0; i < N; i++) 
        {
            double sum = 0.0;
            for(int j = 0; j < M; j++) 
            {
                sum += A_host(i,j);
            }
            rowSums_serial_host(i) = sum;
        }

        double serial_time = timer_serial.seconds();

        Kokkos::deep_copy(rowSums_serial, rowSums_serial_host);

        auto parallel_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rowSums_parallel);

        bool ok = true;
        for(int i = 0; i < N; i++) 
        {
            if (parallel_host(i) != rowSums_serial_host(i)) 
            {
                ok = false;
                break;
            }
        }

        std::cout << "Correct: " << (ok ? "Yes" : "No") << "\n";
        std::cout << "Parallel time: " << parallel_time << " seconds\n";
        std::cout << "Serial time:   " << serial_time   << " seconds\n";
    }
    Kokkos::finalize();
    return 0;
}
