#!/bin/bash
# This script will create new lexicon based on already existing one and provided text corpus
# usage is <existing lexicon> <new corpus of text data>

lexicon=$1
corpus=$2

../local/phonetic_transcription_cs.pl $corpus lexicon_new.txt

python3 merge_lexicons.py $lexicon lexicon_new.txt lexicon_merged.txt lexicon_additional.txt
