#!/bin/bash

corpus=ONDRA_OVERFIT
lexicon_original=lexicon_original.txt

grep -oE "[A-Za-zĚŠČŘŽÝÁÍÉŮÚŤĎŇÓěščřžýáíéúůťďňó\\-\\]{3,}" $corpus | tr '[:lower:]' '[:upper:]' | sort | uniq > words_new.txt

cat words_new.txt | python3 preprocessme.py >> preprocessed_words.txt ; rm words_new.txt

./extend_lexicon.sh $lexicon_original preprocessed_words.txt
