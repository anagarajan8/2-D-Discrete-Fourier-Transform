start=`date +%s%N`
./p33 forward ../input/Tower4.txt ../output/cuda-forward-4.txt 
end=`date +%s%N`
echo '4' $((end-start)) > ../time/time-cuda-forward.txt

start=`date +%s%N`
./p33 forward ../input/Tower8.txt ../output/cuda-forward-8.txt
end=`date +%s%N`
echo '8' $((end-start)) >> ../time/time-cuda-forward.txt

start=`date +%s%N`
./p33 forward ../input/Tower128.txt ../output/cuda-forward-128.txt
end=`date +%s%N`
echo '128' $((end-start)) >> ../time/time-cuda-forward.txt

start=`date +%s%N`
./p33 forward ../input/Tower256.txt ../output/cuda-forward-256.txt
end=`date +%s%N`
echo '256' $((end-start)) >> ../time/time-cuda-forward.txt

start=`date +%s%N`
./p33 forward ../input/Tower512.txt ../output/cuda-forward-512.txt
end=`date +%s%N`
echo '512' $((end-start)) >> ../time/time-cuda-forward.txt

start=`date +%s%N`
./p33 forward ../input/Tower1024.txt ../output/cuda-forward-1024.txt
end=`date +%s%N`
echo '1024' $((end-start)) >> ../time/time-cuda-forward.txt

start=`date +%s%N`
./p33 forward ../input/Tower2048.txt ../output/cuda-forward-2048.txt
end=`date +%s%N`
echo '2048' $((end-start)) >> ../time/time-cuda-forward.txt

