#!/bin/bash

locdata=$1
train_text=$2
test_text=$3

echo "=== Preparing the vocabulary ..."
touch $locdata/vocab-full-raw.txt $locdata/vocab-full-raw.txt $locdata/vocab-test.txt
if [ "$DICTIONARY" == "build" ]; then
  echo; echo "Building dictionary from train data"; echo
  cut -d' ' -f2- $train_text | tr ' ' '\n' > $locdata/vocab-full-raw.txt
else
  echo; echo "Using predefined dictionary: ${DICTIONARY}"
  echo "Throwing away first 2 rows."; echo
  tail -n +3 $DICTIONARY | cut -f 1 > $locdata/vocab-full-raw.txt
fi

echo '</s>' >> $locdata/vocab-full-raw.txt
echo "Removing from vocabulary _NOISE_, and  all '_' words from vocab-full.txt"
cat $locdata/vocab-full-raw.txt | grep -v '_' | \
  sort -u > $locdata/vocab-full.txt
echo "*** Vocabulary preparation finished!"


echo "Removing from vocabulary _NOISE_, and  all '_' words from vocab-test.txt"
cut -d' ' -f2 $test_text | tr ' ' '\n' | grep -v '_' | sort -u > $locdata/vocab-test.txt
