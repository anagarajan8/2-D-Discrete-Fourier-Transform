# 2-D-Discrete-Fourier-Transform
ECE 6122 Final Project
Read FinalProject.pdf for details

The project folder p3 has the following directory structure:
build - build folder where executables are generated.
input - input folder contains all input files.
output - output folder can be used to hold the result of the Forward and Reverse Fourier Transforms.
scripts - script folder contains scripts which can be used to test correctness and performance of the various Fourier Transform implementations.
src - src folder contains source code given in assignment.
time - time folder contains the time values of the various implementations obtained after running scripts in the scripts folder.
main-cuda.cu - Contains Cuda implementation of Forward and Reverse Fourier Transform.
main-mpi.cc - Contains MPI implementation of Forward(FFT) and Reverse Fourier Transform.
main-threads.cc - Contains C++ threads implementation of Forward and Reverse Fourier Transform.

Modules loaded to test the project:
1. gcc/4.9.0
2. openmpi/1.8
3. cuda/9.1
4. cmake/3.9.1

Steps in running the project:
1. cd p3/build
2. cmake ..
3. make
4. ./p31 forward/reverse [Input File] [Output File]
eg : ./p31 forward/reverse ../input/Tower1024.txt ../output/p31-1024.txt 
5. mpirun -np 8 ./p32 forward/reverse [Input File] [Output File]
6. ./p33 forward/reverse [Input File] [Output File]

Running scripts:
1. Copy any script(except run.sh) in the scripts folder to the build folder.
eg : cp p3/script/script-cuda-forward.sh p3/build/
2. Go to the build directory and run the script using the sh command.
eg : cd p3/build; sh script-cuda-forward.sh
3. The performance measurement of the particular implemenatation will be found in the time directory and the result of the fourier transform 
can be found in the output directory.
eg: cd p3/build; sh script-cuda-forward.sh
Checking the output : cd p3/output. ls cuda-forward-*
Checking the performance : cd p3/time. cat time-cuda-forward.txt
