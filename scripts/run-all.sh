cd ~/p3/build
cp ../scripts/script-cuda-forward.sh .
cp ../scripts/script-mpi-forward.sh .
cp ../scripts/script-threads-forward.sh .
sh script-cuda-forward.sh
echo "CUDA : "
cat ../time/time-cuda-forward.txt
sh script-mpi-forward.sh
echo "MPI : "
cat ../time/time-mpi-forward.txt
sh script-threads-forward.sh
echo "THREADS : "
cat ../time/time-threads-forward.txt
