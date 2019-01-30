start=`date +%s%N`
./p31 reverse ../input/Tower4.txt ../output/threads-reverse-4.txt 
end=`date +%s%N`
echo '4' $((end-start)) > ../time/time-threads-reverse.txt

start=`date +%s%N`
./p31 reverse ../input/Tower8.txt ../output/threads-reverse-8.txt
end=`date +%s%N`
echo '8' $((end-start)) >> ../time/time-threads-reverse.txt

start=`date +%s%N`
./p31 reverse ../input/Tower128.txt ../output/threads-reverse-128.txt
end=`date +%s%N`
echo '128' $((end-start)) >> ../time/time-threads-reverse.txt

start=`date +%s%N`
./p31 reverse ../input/Tower256.txt ../output/threads-reverse-256.txt
end=`date +%s%N`
echo '256' $((end-start)) >> ../time/time-threads-reverse.txt

start=`date +%s%N`
./p31 reverse ../input/Tower512.txt ../output/threads-reverse-512.txt
end=`date +%s%N`
echo '512' $((end-start)) >> ../time/time-threads-reverse.txt

start=`date +%s%N`
./p31 reverse ../input/Tower1024.txt ../output/threads-reverse-1024.txt
end=`date +%s%N`
echo '1024' $((end-start)) >> ../time/time-threads-reverse.txt

start=`date +%s%N`
./p31 reverse ../input/Tower2048.txt ../output/threads-reverse-2048.txt
end=`date +%s%N`
echo '2048' $((end-start)) >> ../time/time-threads-reverse.txt

