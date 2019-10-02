#!/bin/bash
# usage pu mp3 name and desired output name
# Get audio file on input
file_name=$1

# name of dictionary is the current dictionary
new_name=$(pwd)
B="$(cut -d'/' -f9 <<<"$new_name")"

rsync -avz jkratochvil@geri.ms.mff.cuni.cz:/lnet/ms/data/cesky-rozhlas-prepisy/2018-prepisy-od-Monitory-audio/downloaded/$1 ~/Jonas-zaloha/Translate_vypisky/Ceskyrozhlas/ProAProti/final/$B

# convert from MP3 to wav and also ensure the proper audio parameters
ffmpeg -i $1 temp.wav

sox temp.wav -r 16000 -c 1 -b 16 $B.wav ; rm temp.wav

mkdir audio_segments

sed -i "/(PÍSNIČKA)/d" ./final_ali.txt
sed -i "/(ZNĚLKA)/d" ./final_ali.txt

sed -r '/^\s*$/d' final_ali.txt > new_ali.txt

rm final_ali.txt

# final_ali.txt contains the text transcription in the correct format

mv new_ali.txt final_ali.txt

# This does the alignment in unsupervised way
python3 ../first_alignment.py $B.wav

rm $1
