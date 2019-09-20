#!/bin/bash
exp=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/exp
for x in $exp/*/decode* $exp/chain/*/decode*;do
    [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh
done
