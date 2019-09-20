#!/bin/bash

set -euo pipefail

ngram_order=2

lmdir=data/local/lm_train; mkdir -p $lmdir
lm_build=$lmdir/lm_${ngram_order}g_kn.gz

cat data/train_noreseg/text | cut -d' ' -f 2- | sed 's:\[[a-z]*\]::g; s:  : :g; s:^ *::g' | tr -d '~' >$lmdir/corpus_train

cut -d' ' -f 1 data/lang/words.txt | grep -v -e '<eps>' -e '\*\*' -e '#[0-9]' >$lmdir/vocab

ngram-count -text $lmdir/corpus_train -order $ngram_order -vocab $lmdir/vocab \
        -unk -map-unk "[oov]" -kndiscount -interpolate -lm $lm_build

utils/format_lm_sri.sh data/lang $lm_build data/lang_test_lmtrain
