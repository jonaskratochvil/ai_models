#!/bin/bash

## USAGE JUST GO TO DIRECTORY WITH WAV FILES AND FINAL_ALI.TXT AND RUN 
# ../MOVE_FILES.SH AND THE REST SHOULD BE JUST WAITING

name=$(basename "`pwd`")
text="final_ali.txt"

rm $text ; cp ../text_files/$name/$text . 

python3 ../splittext.py $text $name
count=$(ls -l $name* | wc -l)
j=1

echo "Starting directory preparations"
for file in *.wav; do
    echo "Preparing file $j out of $(($count / 2))"
    dir_name="${file%.*}"

    # We will deal with cases A10, A5,...
    cat ${dir_name}.txt | python3 ../numtotext.py > tmp.txt ; rm ${dir_name}.txt
    cat tmp.txt | python3 ../numtotext.py > ${dir_name}.txt ; rm tmp.txt

    # Split text to 12 words, seems to work well with alignment
    #tr '\n' ' ' < ${dir_name}.txt > tmp.txt
    text_file=${dir_name}.txt
    tr '\n' ' ' < ${dir_name}.txt > tmp.txt
    #sed 's/\\n/ /g' $text_file > tmp.txt
    rm ${dir_name}.txt
    mv tmp.txt ${dir_name}.wav.trn

    # Again trim silence from end and beginning of each file
    sox $PWD/$file $PWD/tmp.wav silence 1 0.1 1% reverse silence 1 0.1 1% reverse 
    rm $PWD/$file
    mv $PWD/tmp.wav $PWD/$file

    ((j++))
done
rm -r backup final_ali.txt
echo "Done preparing all files"

