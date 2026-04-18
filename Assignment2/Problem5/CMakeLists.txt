cmake_minimum_required(VERSION 3.22)
project(MyProject)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
)
FetchContent_MakeAvailable(Kokkos)

add_executable(Problem5 Problem5.cpp)
target_link_libraries(Problem5 Kokkos::kokkos)