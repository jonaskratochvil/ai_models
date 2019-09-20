#!/bin/bash

# How many files you want to loop over
# Make these command line arguments
# POZOR bash vykoná vždy i ten poslední!
# On which day it takes place and how many sessions
DAY=$1
MONTH=03
YEAR=2019
MEETING=27
NEW_DIR=snemovna_${MEETING}_${YEAR}_${MONTH}_${DAY}
# Cat the final table with times to array
# j = 0 times[0]!!!
j=0
dir=files

python3 ./get_audiotimes.py $DAY $MONTH $YEAR > table.txt

cat table.txt | python3 get_timetable.py $MEETING > final_table_with_times.txt ; rm table.txt

# This cuts number
cat final_table_with_times.txt  | sed 's/|/ /' | awk '{print $2}' > times.txt

# This cuts times
cat final_table_with_times.txt  | sed 's/|/ /' | awk '{print $1}' > final_table.txt

times=( $(cat times.txt) )
LIMIT=$(cat times.txt | wc -l)
END=$((${times[0]} + $LIMIT - 1))
arr=( $(cat final_table.txt) )
# This is here because we do not want to pad the last audiofile
condition=false
echo ${times[0]}
echo $LIMIT
echo $END

for i in $(seq ${times[0]} $END); do
    # Takes the most recent time from array
    # Here index by j as I take values from array
    audio_time=${arr[$j]}
    echo $audio_time
    echo "======================== Processing file $((j+1)) out of $LIMIT files and day: $DAY ======================="
    # get the corresponding texts
    # Here I want to index by i as I am requesting from website
    # Pozor casy meetingu nemusi byt linearni proto je ber z pole times
    python3 gettext.py ${times[$j]} $MEETING > fnl.txt
    cat fnl.txt | python3 strip.py >> $dir/final.txt
    rm fnl.txt


    # Download the MP3 and save it to dir
    url="https://www.psp.cz/eknih/2017ps/audio/${YEAR}/${MONTH}/${DAY}/${YEAR}${MONTH}${DAY}${audio_time}.mp3"
    wget $url -P $dir
    
    audio="${YEAR}${MONTH}${DAY}${audio_time}.mp3"

    # On last iteration we do not pad the audio
    if (( $i == $LIMIT )) ; then
        condition=true
    fi

    ./audioconcat2.sh $dir/$audio $condition
    ((j++))
done

echo "Done downloading the data, doing data preparation"
wait

# On last iteration we add last file
#python3 gettext.py $(($END +1)) > fnl.txt
#cat fnl.txt | python3 strip.py >> $dir/final.txt
#rm fnl.txt

mv merged* files/

rm files/*.mp3

# Takes everything inside brackets away
sed -e 's/([^()]*)//g' files/final.txt > files/final_nobrckt.txt

# Do various text manipulations, it is neccesary to run it twice because of time things like
# 12.12 which are split first time and converted to text next time
cat files/final_nobrckt.txt | python3 numtotext.py > tmp.txt ; rm files/final_nobrckt.txt
cat tmp.txt | python3 numtotext.py > tmp_secondround.txt
# Tohle to asi ocisti? Do split on .,?!
sed 's/[.!?,] */&\n/g' ./tmp_secondround.txt > tmp1.txt
tr -d \? < tmp1.txt > tmp2.txt
tr -d \. < tmp2.txt > outfile.txt
tr -d \! < outfile.txt > outfile2.txt
tr -d \, < outfile2.txt > outfile3.txt
# This takes away the weird like ends
tr -d $'\r' < outfile3.txt > ll.txt

cat -s ll.txt > final_ali.txt

mv final_ali.txt files/

# Make new dir according to session number, move relevant files there and split merged wav in 15
# minutes long chunks
mkdir -p files/$NEW_DIR files/$NEW_DIR/backup

rm tmp.txt tmp1.txt tmp2.txt outfile2.txt outfile.txt ll.txt tmp_secondround.txt outfile3.txt

mv files/merged* files/merged.wav

# This should trim silence from both beginning and end of audio file
sox files/merged.wav files/merged_no_silence.wav silence 1 0.1 1% reverse silence 1 0.1 1% reverse

rm files/merged.wav final_table.txt final_table_with_times.txt times.txt

mv files/final.txt files/$NEW_DIR/backup

mv files/final_ali.txt files/merged_no_silence.wav files/$NEW_DIR/

sox files/$NEW_DIR/merged_no_silence.wav files/$NEW_DIR/$NEW_DIR.wav trim 0 1800 : newfile : restart

rm files/$NEW_DIR/merged_no_silence.wav 

wait

echo "Done with data preparation moving files to flash drive"

#mv files/* /media/jonas/72D7-4FB6/Poslanecka_snemovna/

echo "Done with the whole damn thing"
