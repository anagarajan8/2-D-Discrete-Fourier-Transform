start=`date +%s%N`
mpirun -np 8 ./p32 reverse ../input/Tower4.txt ../output/mpi-reverse-4.txt
end=`date +%s%N`
echo '4' $((end-start)) > ../time/time-mpi-reverse.txt

start=`date +%s%N`
mpirun -np 8 ./p32 reverse ../input/Tower8.txt ../output/mpi-reverse-8.txt
end=`date +%s%N`
echo '8' $((end-start)) >> ../time/time-mpi-reverse.txt

start=`date +%s%N`
mpirun -np 8 ./p32 reverse ../input/Tower128.txt ../output/mpi-reverse-128.txt
end=`date +%s%N`
echo '128' $((end-start)) >> ../time/time-mpi-reverse.txt

start=`date +%s%N`
mpirun -np 8 ./p32 reverse ../input/Tower256.txt ../output/mpi-reverse-256.txt
end=`date +%s%N`
echo '256' $((end-start)) >> ../time/time-mpi-reverse.txt

start=`date +%s%N`
mpirun -np 8 ./p32 reverse ../input/Tower512.txt ../output/mpi-reverse-512.txt
end=`date +%s%N`
echo '512' $((end-start)) >> ../time/time-mpi-reverse.txt

start=`date +%s%N`
mpirun -np 8 ./p32 reverse ../input/Tower1024.txt ../output/mpi-reverse-1024.txt
end=`date +%s%N`
echo '1024' $((end-start)) >> ../time/time-mpi-reverse.txt

start=`date +%s%N`
mpirun -np 8 ./p32 reverse ../input/Tower2048.txt ../output/mpi-reverse-2048.txt
end=`date +%s%N`
echo '2048' $((end-start)) >> ../time/time-mpi-reverse.txt
