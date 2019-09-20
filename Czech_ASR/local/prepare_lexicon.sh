#!/bin/bash

set -eu

[ $# != 1 ] && echo "Usage: $0 <speechdat-dir>" && exit 1

dir=$1

lexicon=$dir/table/lexicon.tbl

# format the lexicon to kaldi format,
dict=data/local/dict; mkdir -p $dict
tail -n+2 $lexicon | \
  uconv -f 8859-2 -t utf8 | \
  awk -v FS='\t' '{ for (n=3; n<=NF; n++) { print $1, $(n) }; }' | \
  sed 's:\r::g' \
  >$dict/tmp_lexicon.txt

# map the 'words' to lowecase (phonemes stay same),
paste <(awk '{ print $1 }' $dict/tmp_lexicon.txt | uconv -f utf8 -t utf8 -x "Any-Lower") \
      <(cut -d' ' -f2- $dict/tmp_lexicon.txt | sed 's:a_u:au:g; s:o_u:ou:g; s:e_u:eu:g; s:d_z:dz:g; s:d_Z:dZ:g; s:t_s:ts:g; s:t_S:tS:g;') | \
      sort | uniq >$dict/lexicon.txt

# append the 'extra words',
{ echo "[sta] noi"
  echo "[spk] gbg"
  echo "[int] noi"
  echo "[fil] sil"
  echo "[oov] gbg"
  echo "** gbg"
} >>$dict/lexicon.txt

# ---

# extract the phoneme set,
cat $dict/lexicon.txt | awk '{ $1=""; print $0; }' | tr ' ' '\n' | \
  sort | uniq |  egrep -v 'sil|noi|gbg' | awk '$1' >$dict/nonsilence_phones.txt

echo "sil
noi
gbg" >$dict/silence_phones.txt

echo "sil" >$dict/optional_silence.txt

