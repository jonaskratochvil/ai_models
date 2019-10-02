# Get rid of all lines which include names passed as argument
#!/bin/bash

# Only mandatory input is text file name
file_name=$1
first_name=$2
second_name=$3
third_name=$4

sed -i "/$first_name/d" ./$file_name
sed -i "/$second_name/d" ./$file_name
if [ $# -eq 4 ]; then
  sed -i "/$third_name/d" ./$file_name
fi
# Remove first 7 lines as they include some metadata information not which are not important for us
tail -n +7 $file_name > removedlines.txt
# The python script does some minor preprocessing and capitalization of the output
cat removedlines.txt | python3 makepretty.py > tmp.txt
# After symbols ?!. we put new line
sed 's/[.!?] */&\n/g' ./tmp.txt > tmp1.txt
tr -d \? < tmp1.txt > tmp2.txt
tr -d \. < tmp2.txt > outfile.txt
tr -d \! < outfile.txt > outfile2.txt
# Get rid of the trailing end of sentence
tr -d $'\r' < outfile2.txt > ll.txt

cat -s ll.txt > final_ali.txt