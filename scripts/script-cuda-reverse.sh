start=`date +%s%N`
./p33 reverse ../input/Tower4.txt ../output/cuda-reverse-4.txt 
end=`date +%s%N`
echo '4' $((end-start)) > ../time/time-cuda-reverse.txt

start=`date +%s%N`
./p33 reverse ../input/Tower8.txt ../output/cuda-reverse-8.txt
end=`date +%s%N`
echo '8' $((end-start)) >> ../time/time-cuda-reverse.txt

start=`date +%s%N`
./p33 reverse ../input/Tower128.txt ../output/cuda-reverse-128.txt
end=`date +%s%N`
echo '128' $((end-start)) >> ../time/time-cuda-reverse.txt

start=`date +%s%N`
./p33 reverse ../input/Tower256.txt ../output/cuda-reverse-256.txt
end=`date +%s%N`
echo '256' $((end-start)) >> ../time/time-cuda-reverse.txt

start=`date +%s%N`
./p33 reverse ../input/Tower512.txt ../output/cuda-reverse-512.txt
end=`date +%s%N`
echo '512' $((end-start)) >> ../time/time-cuda-reverse.txt

start=`date +%s%N`
./p33 reverse ../input/Tower1024.txt ../output/cuda-reverse-1024.txt
end=`date +%s%N`
echo '1024' $((end-start)) >> ../time/time-cuda-reverse.txt

start=`date +%s%N`
./p33 reverse ../input/Tower2048.txt ../output/cuda-reverse-2048.txt
end=`date +%s%N`
echo '2048' $((end-start)) >> ../time/time-cuda-reverse.txt

