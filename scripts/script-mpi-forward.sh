start=`date +%s%N`
mpirun -np 8 ./p32 forward ../input/Tower4.txt ../output/mpi-forward-4.txt
end=`date +%s%N`
echo '4' $((end-start)) > ../time/time-mpi-forward.txt

start=`date +%s%N`
mpirun -np 8 ./p32 forward ../input/Tower8.txt ../output/mpi-forward-8.txt
end=`date +%s%N`
echo '8' $((end-start)) >> ../time/time-mpi-forward.txt

start=`date +%s%N`
mpirun -np 8 ./p32 forward ../input/Tower128.txt ../output/mpi-forward-128.txt
end=`date +%s%N`
echo '128' $((end-start)) >> ../time/time-mpi-forward.txt

start=`date +%s%N`
mpirun -np 8 ./p32 forward ../input/Tower256.txt ../output/mpi-forward-256.txt
end=`date +%s%N`
echo '256' $((end-start)) >> ../time/time-mpi-forward.txt

start=`date +%s%N`
mpirun -np 8 ./p32 forward ../input/Tower512.txt ../output/mpi-forward-512.txt
end=`date +%s%N`
echo '512' $((end-start)) >> ../time/time-mpi-forward.txt

start=`date +%s%N`
mpirun -np 8 ./p32 forward ../input/Tower1024.txt ../output/mpi-forward-1024.txt
end=`date +%s%N`
echo '1024' $((end-start)) >> ../time/time-mpi-forward.txt

start=`date +%s%N`
mpirun -np 8 ./p32 forward ../input/Tower2048.txt ../output/mpi-forward-2048.txt
end=`date +%s%N`
echo '2048' $((end-start)) >> ../time/time-mpi-forward.txt
