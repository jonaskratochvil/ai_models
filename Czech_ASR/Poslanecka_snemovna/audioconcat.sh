#!/bin/bash

file1=$1
file2=$2
file3=$3

< /dev/null ffmpeg -ss 00:00:00 -t 00:10:00 -i $file1 -acodec copy tmp1.mp3
< /dev/null ffmpeg -ss 00:00:00 -t 00:10:00 -i $file2 -acodec copy tmp2.mp3
< /dev/null ffmpeg -ss 00:00:00 -t 00:10:00 -i $file3 -acodec copy tmp3.mp3

< /dev/null ffmpeg -i tmp1.mp3 output1.wav ; rm tmp1.mp3
    
# convert wav file to match recordings
sox output1.wav -r 16000 -c 1 output1_new.wav ; rm output1.wav

< /dev/null ffmpeg -i tmp2.mp3 output2.wav ; rm tmp2.mp3
    
# convert wav file to match recordings
sox output2.wav -r 16000 -c 1 output2_new.wav ; rm output2.wav

< /dev/null ffmpeg -i tmp3.mp3 output3.wav ; rm tmp3.mp3
    
# convert wav file to match recordings
sox output3.wav -r 16000 -c 1 output3_new.wav ; rm output3.wav

sox output1_new.wav output2_new.wav output3_new.wav merged.wav ; rm output1_new.wav output2_new.wav output3_new.wav

sox merged.wav merged_final.wav silence -l 1 0.1 1% -1 2.0 1% ; rm merged.wav

