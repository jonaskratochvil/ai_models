#!/bin/bash
# Bugs fixed so far: compile Kaldi with CUDA, install SRILM,   lower number of decoding jobs, detect corrupted audio files
# IN data need train dev and local!!!!



set -euxo pipefail # exit on error, enable debug mode,

echo "======================== Starting Rozhlas data run script =============="
data=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/new_data/train/experiment_dir

. ./env_voip_cs.sh
. ./cmd.sh
. ./path.sh
#. utils/parse_options.sh

nj_dec=1
lm="build3"
stage=5
jobs_final=3
jobs_initial=1
chunk_per_minibatch="256,128,64"

# Do the data preparation,
echo "======================= Starting data preparation ======================"
if [ $stage -le 2 ]; then
    local/data_split.sh --every_n 1 $data data "$lm" "dev"

    # This should be here
    #local/prepare_dictionary.sh data/local data/train/trans.txt data/dev/trans.txt


    local/create_LMs.sh data/local data/train/trans.txt \
     data/test/trans.txt data/local/lm "$lm"

    gzip data/local/lm/$lm

    local/prepare_cs_transcription.sh data/local data/local/dict

    local/create_phone_lists.sh data/local/dict

    utils/prepare_lang.sh data/local/dict '_SIL_' data/local/lang data/lang

    for part in train dev; do
        file=data/$part/trans.txt
        if [ -f $file ]; then
            mv data/$part/trans.txt data/$part/text
        fi
    done

    #local/prepare_lm.sh

    utils/format_lm.sh data/lang data/local/lm/$lm.gz data/local/dict/lexicon.txt data/lang_test

    utils/fix_data_dir.sh data/train
    utils/fix_data_dir.sh data/dev

    for part in train dev; do
        utils/validate_data_dir.sh --no-feats data/$part
    done
fi

echo "======================== Feature extraction ==========================="

mfccdir=mfccs
if [ $stage -le 2 ]; then
    for part in train; do
        #utils/validate_data_dir.sh data/$part
        steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
        steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
    done

    steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 data/dev exp/make_mfcc/dev $mfccdir
    steps/compute_cmvn_stats.sh data/dev exp/make_mfcc/dev $mfccdir
fi

echo "======================== Training Monophone     system ====================="
if [ $stage -le 3 ]; then
    utils/subset_data_dir.sh --shortest data/train 15000 data/train_15kshort

    steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
        data/train_15kshort data/lang exp/mono

    steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd  "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali
fi
#echo "======================== Training Triphone1 system ====================="
if [ $stage -le 4 ]; then
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
        data/train data/lang_big exp/mono_ali exp/tri1

    utils/mkgraph.sh data/lang_big exp/tri1 exp/tri1/graph

    steps/decode.sh --nj $nj_dec --cmd "$decode_cmd" \
        exp/tri1/graph data/dev exp/tri1/decode_dev

    steps/align_si.sh --nj 10 --cmd "$train_cmd" \
        data/train data/lang exp/tri1 exp/tri1_ali
fi
# tri2,
#echo "======================== Training Triphone2 system ====================="
if [ $stage -le 5 ]; then
    steps/train_lda_mllt.sh --cmd "$train_cmd" \
           --splice-opts "--left-context=3 --right-context=3" 4200 40000 \
           data/train data/lang_big exp/tri1_ali exp/tri2

    # This means that it can run in background if you put ()&

    utils/mkgraph.sh data/lang_big exp/tri2 exp/tri2/graph

    # Karel said to decrease nj for decoding

    steps/decode.sh --nj $nj_dec --cmd "$decode_cmd" \
        exp/tri2/graph data/dev exp/tri2/decode_dev


    steps/align_si.sh --nj 10 --cmd "$train_cmd" \
        data/train data/lang exp/tri2 exp/tri2_ali

    steps/get_train_ctm.sh --use-segments false --print-silence true \
        data/train data/lang exp/tri2_ali
fi

if [ $stage -le 1 ]; then
    steps/train_sat.sh --cmd "$train_cmd" \
    #          3000 30000 data/train data/lang_big exp/tri2_ali exp/tri3

    utils/mkgraph.sh data/lang_big exp/tri3 exp/tri3/graph

    steps/decode_fmllr.sh --nj $nj_dec --cmd "$decode_cmd" \
               exp/tri3/graph data/dev exp/tri3/decode_dev
fi

if [ $stage -le 1 ]; then
    # The following script cleans the data and produces cleaned data
    steps/cleanup/clean_and_segment_data.sh --nj 10 --cmd "$train_cmd" \
        --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
        data/train data/lang exp/tri3 exp/tri3_cleaned data/train_cleaned
fi
nnet_stage=10
if [ $stage -le 1 ]; then

    local/chain/run_tdnn.sh
    #--nj 10 --stage $nnet_stage --train-set train_cleaned --test-sets "dev" \
    #    --gmm tri3_cleaned --nnet3-affix _train_cleaned_rvb --chunk_per_minibatch "$chunk_per_minibatch" --jobs_initial $jobs_initial --jobs_final $jobs_final
fi
