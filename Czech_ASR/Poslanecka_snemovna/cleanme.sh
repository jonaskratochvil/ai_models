#!/bin/bash

# Only mandatory input is text file name
file_name=$1

sed -e 's/([^()]*)//g' files/final.txt > files/final_nobrckt.txt
sed -i "/Zdroj:/d" ./$file_name
sed -i "/Datum vysílání:/d" ./$file_name
sed -i "/Čas vysílání:/d" ./$file_name
sed -i "/pořad:/d" ./$file_name
sed -i "/Stopáž:/d" ./$file_name
sed -i "/Pořadí:/d" ./$file_name
sed -i "/<<<Konec>>>/d" ./$file_name
sed -i "/Datum:/d" ./$file_name
sed -i "/Název:/d" ./$file_name
sed -i "/Rozsah:/d" ./$file_name
sed -i "/Text:/d" ./$file_name

cat $file_name | python3 makepretty.py > tmp.txt
sed 's/[.!?] */&\n/g' ./tmp.txt > tmp1.txt
tr -d \? < tmp1.txt > tmp2.txt
tr -d \. < tmp2.txt > outfile.txt
tr -d \! < outfile.txt > outfile2.txt

tr -d $'\r' < outfile2.txt > ll.txt

cat -s ll.txt > final_ali.txt

rm tmp.txt tmp1.txt tmp2.txt outfile2.txt outfile.txt ll.txt
