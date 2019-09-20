#!/bin/bash
file1=$1
cond=$2

if [ "$cond" = false ] ; then
    < /dev/null ffmpeg -ss 00:00:00 -t 00:10:00 -i $file1 -acodec copy tmp1.mp3

else
    mv $file1 tmp1.mp3
fi



< /dev/null ffmpeg -i tmp1.mp3 output1.wav ; rm tmp1.mp3
    
# convert wav file to match recordings
sox output1.wav -r 16000 -c 1 output1_new.wav ; rm output1.wav

if [[ -f merged.wav ]] ; then

  sox merged.wav output1_new.wav merged_new.wav; rm output1_new.wav merged.wav

elif [[ -f merged_new.wav ]] ; then

  sox merged_new.wav output1_new.wav merged.wav; rm output1_new.wav merged_new.wav

else

  sox output1_new.wav output1_silence.wav silence -l 1 0.1 1% -1 2.0 1%

  mv output1_silence.wav merged.wav

fi