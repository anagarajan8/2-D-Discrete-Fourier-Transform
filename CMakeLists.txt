cmake_minimum_required(VERSION 3.9)

find_package (Threads)
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Werror -g")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

include_directories(
    ./src
)

set(SOURCE_FILES
    ./src/complex.h
    ./src/input_image.h
    ./src/complex.cc
    ./src/input_image.cc
)

set(SOURCE_FILES_1
    ./src/complex.cuh
    ./src/input_image.h
    ./src/complex.cu
    ./src/input_image.cc
)

add_executable(p31 ${SOURCE_FILES} main-threads.cc)
target_link_libraries (p31 ${CMAKE_THREAD_LIBS_INIT})

add_executable(p32 ${SOURCE_FILES} main-mpi.cc)
target_link_libraries(p32 ${MPI_LIBRARIES})

SET (CMAKE_STANDARD 17)
SET(CUDA_SEPARABLE_COMPILATION ON)
SET (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
CUDA_ADD_EXECUTABLE(p33 ${SOURCE_FILES_1} main-cuda.cu)
