#!/bin/bash

## USAGE JUST GO TO DIRECTORY WITH WAV FILES AND FINAL_ALI.TXT AND RUN 
# ../MOVE_FILES.SH AND THE REST SHOULD BE JUST WAITING

name=$(basename "`pwd`")
text="final_ali.txt"

cp -r /home/jonas/Jonas-zaloha/Translate_vypisky/Ceskyrozhlas/Poslanecka_Snemovna/files/${name} \
/home/jonas/Jonas-zaloha/Translate_vypisky/Ceskyrozhlas/Poslanecka_Snemovna/files/${name}_bckp

python3 ../splittext.py $text $name
count=$(ls -l $name* | wc -l)
j=1

echo "Starting directory preparations"
for file in *.wav; do
    echo "Preparing directory $j out of $(($count / 2))"
    dir_name="${file%.*}"
    mkdir -p $dir_name
    mkdir -p $dir_name/audio_segments

    # We will deal with cases A10, A5,...
    cat ${dir_name}.txt | python3 ../numtotext.py > tmp.txt ; rm ${dir_name}.txt
    cat tmp.txt | python3 ../numtotext.py > ${dir_name}.txt ; rm tmp.txt

    # Split text to 12 words, seems to work well with alignment
    #tr '\n' ' ' < ${dir_name}.txt > tmp.txt
    text_file=${dir_name}.txt
    sed 's/\\n/ /g' $text_file > tmp.txt
    xargs -n12 < tmp.txt > tmp2.txt
    rm ${dir_name}.txt tmp.txt
    mv tmp2.txt ${dir_name}.txt

    # Again trim silence from end and beginning of each file
    sox $PWD/$file $PWD/tmp.wav silence 1 0.1 1% reverse silence 1 0.1 1% reverse 
    rm $PWD/$file
    mv $PWD/tmp.wav $PWD/$file

    # Do the alignment
    mv $file ${dir_name}.txt $dir_name
    python3 ../second_alignment.py $name $dir_name
    rm -r $dir_name/output $dir_name/${file} $dir_name/*.txt
    ((j++))
done

mkdir ../text_files/${name}
rm -r backup
mv $text ../text_files/${name}

echo "Done preparing all directorias"

