#!/bin/bash

set -euxo pipefail # exit on error, enable debug mode,

data=$HOME/personal_work_ms/00/rozhlas_data/new_data/experiment_dir/

. ./env_voip_cs.sh
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

nj_dec=8
lm="build3"
stage=1

if [ $stage -le 1 ]; then

    local/data_split.sh --every_n 1 $data data "$lm" "dev"

    local/prepare_dictionary.sh data/local data/train/trans.txt data/dev/trans.txt

    local/prepare_cs_transcription.sh data/local data/local/dict

    local/create_phone_lists.sh data/local/dict

    utils/prepare_lang.sh data/local/dict '_SIL_' data/local/lang data/lang

    # Move trans.txt to text to build LM latter
    for part in train dev; do
        file=data/$part/trans.txt
        if [ -f $file ]; then
            mv data/$part/trans.txt data/$part/text
        fi
    done
fi

if [ $stage -le 2 ]; then
    # Prepare language model at this stage from domain specific texts and general texts
    local/prepare_lm.sh

    utils/fix_data_dir.sh data/train
    utils/fix_data_dir.sh data/dev

    for part in train dev; do
        utils/validate_data_dir.sh --no-feats data/$part
    done
fi

# TODO add noises from RIR(?) and augment data this way, do you do that after some training or right here?
# Inspire from chime5/s5b and train GMM on all -> run the data clean up script before TDNN


if [ $stage -le 8 ]; then
    # Now make MFCC features.
    # mfccdir should be some place with a largish disk where you
    # want to store MFCC features.
    mfccdir=mfcc # Put them probably somewhere to /lnet/ms/data/rozhlas...
    for x in ${train_set} ${test_sets}; do
        steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
                data/$x exp/make_mfcc/$x $mfccdir
        steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
        utils/fix_data_dir.sh data/$x
    done
fi

if [ $stage -le 9 ]; then
    # make a subset for monophone training
    utils/subset_data_dir.sh --shortest data/${train_set} 100000 data/${train_set}_100kshort
    utils/subset_data_dir.sh data/${train_set}_100kshort 30000 data/${train_set}_30kshort
fi

if [ $stage -le 10 ]; then
    # Starting basic training on MFCC features
    steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
               data/${train_set}_30kshort data/lang exp/mono
fi

if [ $stage -le 11 ]; then
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
             data/${train_set} data/lang exp/mono exp/mono_ali

    steps/train_deltas.sh --cmd "$train_cmd" \
             2500 30000 data/${train_set} data/lang exp/mono_ali exp/tri1
fi



