#!/bin/bash
# Bugs fixed so far: compile Kaldi with CUDA, install SRILM, lower number of decoding jobs, detect corrupted audio files
set -euxo pipefail # exit on error, enable debug mode,

echo "======================== Starting Rozhlas data run script =============="
data=$HOME/personal_work_ms/00/rozhlas_data/new_data/experiment_dir/

. ./env_voip_cs.sh
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

nj_dec=8
lm="build3"
# Do the data preparation,
echo "======================= Starting data preparation ======================"

local/data_split.sh --every_n 1 $data data "$lm" "dev"

local/create_LMs.sh data/local data/train/trans.txt \
data/test/trans.txt data/local/lm "$lm"

gzip data/local/lm/$lm

local/prepare_cs_transcription.sh data/local data/local/dict

local/create_phone_lists.sh data/local/dict

utils/prepare_lang.sh data/local/dict '_SIL_' data/local/lang data/lang

utils/format_lm.sh data/lang data/local/lm/$lm.gz data/local/dict/lexicon.txt \
data/lang_test

utils/fix_data_dir.sh data/train

for part in dev train; do
    mv data/$part/trans.txt data/$part/text
    utils/validate_data_dir.sh --no-feats data/$part
    utils/fix_data_dir.sh data/$part

    echo "done moving $part "
done

echo "======================== Feature extraction ==========================="

#
# Begin the recipe,
#
mfccdir=mfccs
utils/fix_data_dir.sh data/train

for part in dev train; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
done

utils/subset_data_dir.sh --shortest data/train 1000 data/train_1kshort

# mono,
echo "======================== Training Monophone system ====================="

steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
  data/train_1kshort data/lang exp/mono

steps/align_si.sh --boost-silence 1.25 --nj 30 --cmd "$train_cmd" \
  data/train data/lang exp/mono exp/mono_ali

# tri1,
echo "======================== Training Triphone1 system ====================="

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
  data/train data/lang exp/mono_ali exp/tri1

utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph

steps/decode.sh --nj $nj_dec --cmd "$decode_cmd" \
  exp/tri1/graph data/dev exp/tri1/decode_dev

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train data/lang exp/tri1 exp/tri1_ali

# tri2,
echo "======================== Training Triphone2 system ====================="

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" 4200 40000 \
  data/train data/lang exp/tri1_ali exp/tri2

utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
# Karel said to decrease nj for decoding
steps/decode.sh --nj $nj_dec --cmd "$decode_cmd" \
  exp/tri2/graph data/dev exp/tri2/decode_dev

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train data/lang exp/tri2 exp/tri2_ali

steps/get_train_ctm.sh --use-segments false --print-silence true \
  data/train data/lang exp/tri2_ali
# creates exp/tri2_ali/ctm

# chain,
echo "======================== Training chain model ========================="
local/chain/run_tdnn.sh


#---------------------------- Decode with 'train_lm' from training transcripts,
utils/mkgraph.sh data/lang_test_lmtrain exp/tri2 exp/tri2/graph_lmtrain

steps/decode.sh --nj $nj_dec --cmd "$decode_cmd" \
  exp/tri2/graph_lmtrain data/dev exp/tri2/decode_dev_lmtrain

wait

# Find best WER
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

