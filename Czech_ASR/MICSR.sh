#!/bin/bash
# Bugs fixed so far: compile Kaldi with CUDA, install SRILM,   lower number of decoding jobs, detect corrupted audio files
# IN data need train dev and local!!!!
# Quite nice receipt that probably I should adapt: https://github.com/uhh-lt/kaldi-tuda-de/blob/master/s5_r2/run.sh
set -euxo pipefail # exit on error, enable debug mode,

. ./env_voip_cs.sh
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

echo "======================== Starting MICSR data run script =============="

# Data directories (beware that for large models data + exp directory can take up to 200G)
data=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/data
data_dir=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/new_data/
exp=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/exp
mfccdir=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/mfccs

# HMM-GMM parameters
nj_dec=6
nj=10
lm="build3"
augmentation=false
combine=false
train=train

# DNN parameters
stage=2
jobs_final=3
jobs_initial=1
chunk_per_minibatch="256,128,64"
online_dir=exp/tdnn_sp_online
online=false

# Do the data preparation,
if [ $stage -le 1 ]; then
    echo "======================= Starting data preparation ======================"

    # BEFORE YOU DO ANYTHING CHECK IF THE DATA DIRECTORY CONTAINS train, local and test subdirectories
    # ALSO MAKE SURE THAT YOU HAVE EXTENDED THE SILENCE IN WAV FILES

    local/data_split.sh --every_n 1 $data_dir $data "$lm" "test_OV test_PS test_rozhlas test_wgvat"

    # This should be here
    # I should at this point put all text from LM to prepare dictionary script
    #local/prepare_dictionary.sh data/local data/train/trans.txt data/dev/trans.txt

    local/create_LMs.sh $data/local $data/train/trans.txt \
     $data/test_rozhlas/trans.txt $data/local/lm "$lm"

    gzip $data/local/lm/$lm
    # This is not neccessary as it is enough to import own dictionary probably, the only additional thing happening here is that it
    # cerates oov_words.txt ?
    local/prepare_cs_transcription.sh $data/local $data/local/dict

    local/create_phone_lists.sh $data/local/dict

    utils/prepare_lang.sh $data/local/dict '_SIL_' $data/local/lang $data/lang

    for part in train test_OV test_PS test_rozhlas test_wgvat; do
        file=$data/$part/trans.txt
        if [ -f $file ]; then
            mv $data/$part/trans.txt $data/$part/text
        fi
    done

    # During training use the LM from transcriptions

    #local/prepare_lm.sh

    utils/format_lm.sh $data/lang $data/local/lm/$lm.gz $data/local/dict/lexicon.txt $data/lang_test

    for part in train test_OV test_PS test_rozhlas test_wgvat; do
        utils/fix_data_dir.sh $data/$part
        utils/validate_data_dir.sh --no-feats $data/$part
    done
fi

# TODO Probably here add somehow data preparation for Poslanecká sněmovna

# Somewhere here I will do the combination of Poslanecká snemovna data and the rest, it is good that I can have utt2uniq in one set of data (this is what I have been using so far) and another set such as Poslanecká sněmovna with just segment file, which I will get by segment_long_utterances_nnet2.sh

if [ "$combine" == true ]; then
    # If I want to combine data
    utils/combine_data.sh <target_dir> <src1_dir> <src2_dir>

    # change train = target_dir
fi

if [ "$augmentation" == true ]; then
    echo "======================== Doing data augmentation  ====================="
    # Inspire from chime5/s5b and train GMM on all -> run the data clean up script before TDNN
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

    steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 1 \
        --isotropic-noise-addition-probability 1 \
        --num-replications 1 \
        --max-noises-per-minute 1 \
        --source-sampling-rate 16000 \
        $data/$train $data/train_reverb

    # Here we add suffix to reverberated utterances

    utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
    rm -rf data/train_reverb
    mv data/train_reverb.new data/train_reverb

    train=train_reverb
fi

if [ $stage -le 1 ]; then
    echo "======================== Making MFCC features  ========================="
    # Better be fixing multiple times here
    for part in $train test_OV test_PS test_rozhlas test_wgvat; do
        utils/fix_data_dir.sh $data/$part
        steps/make_mfcc.sh --cmd "$mfcc_cmd" --nj $nj $data/$part $exp/make_mfcc/$part $mfccdir
        utils/fix_data_dir.sh $data/$part
        steps/compute_cmvn_stats.sh $data/$part $exp/make_mfcc/$part $mfccdir
        utils/fix_data_dir.sh $data/$part
    done
fi
if [ $stage -le 1 ]; then

    # Subset 80000 shortest and from them pick randomly 30000 to that we do not have only
    # Uh, Ah,...
    utils/subset_data_dir.sh --shortest $data/$train 80000 $data/train_80kshort
    utils/subset_data_dir.sh $data/train_80kshort 30000 $data/train_30kshort

    # We will not use the 80k in the future
    rm -r $data/train_80kshort

    # Now subset first 100k which we will use for before training on whole dataset
    utils/subset_data_dir.sh $data/$train 100000 $data/train_100k

fi
if [ $stage -le 1 ]; then
    echo "======================== Training Monophone system ====================="
    steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$mfcc_cmd" \
    $data/train_30kshort $data/lang $exp/mono

    # Alignment is a sequence of integers for each utterance representing alignments for
    # each utterance - it is a collection of transition probabilities, info about phonemes
    # and some other stuff basically all sufficient statistics needed for update

    steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd  "$mfcc_cmd" \
    $data/train_100k $data/lang $exp/mono $exp/mono_ali
fi

if [ $stage -le 1 ]; then
    # swbd 3200 30000
    echo "======================== Training delta system ========================="
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$mfcc_cmd" 2000 10000 \
        $data/train_100k $data/lang_test $exp/mono_ali $exp/tri1

    utils/mkgraph.sh $data/lang_test $exp/tri1 $exp/tri1/graph

    for part in test_OV test_PS test_rozhlas test_wgvat; do
        steps/decode.sh --nj $nj_dec --cmd "$decode_cmd" \
            $exp/tri1/graph $data/$part $exp/tri1/decode_$part
    done

    # Now we start using full data (test how this all works with these subsetting of data)
    steps/align_si.sh --nj $nj --cmd "$mfcc_cmd" \
        $data/$train $data/lang $exp/tri1 $exp/tri1_ali
fi

# tri1
if [ $stage -le 1 ]; then
    # swbd 3200 30000
    echo "======================== Training delta+delta system ==================="
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$mfcc_cmd" 2500 15000 \
        $data/$train $data/lang_test $exp/tri1_ali $exp/tri2

    utils/mkgraph.sh $data/lang_test $exp/tri1 $exp/tri2/graph

    for part in test_OV test_PS test_rozhlas test_wgvat; do
        steps/decode.sh --nj $nj_dec --cmd "$decode_cmd" \
            $exp/tri2/graph $data/$part $exp/tri2/decode_$part
    done

    steps/align_si.sh --nj $nj --cmd "$mfcc_cmd" \
        $data/$train $data/lang $exp/tri2 $exp/tri2_ali
fi

# tri2,
if [ $stage -le 1 ]; then
    echo "=============================== Training FMLLR ========================="

    # FMLLR adapts acoustic model through feature transformation
    steps/train_lda_mllt.sh --cmd "$mfcc_cmd" \
           --splice-opts "--left-context=3 --right-context=3" 3500 20000 \
           $data/$train $data/lang_test $exp/tri2_ali $exp/tri3

    # This means that it can run in background if you put ()&

    utils/mkgraph.sh $data/lang_test $exp/tri3 $exp/tri3/graph

    # Karel said to decrease nj for decoding

    for part in test_OV test_PS test_rozhlas test_wgvat; do
         steps/decode.sh --nj $nj_dec --cmd "$decode_cmd" \
             $exp/tri3/graph $data/$part $exp/tri3/decode_$part
    done

    steps/align_si.sh --nj $nj --cmd "$mfcc_cmd" \
        $data/$train $data/lang $exp/tri3 $exp/tri3_ali
fi

if [ $stage -le 1 ]; then
    echo "=========================== Training SAT model  ========================"

    # 11500 200000 check with the german github above the number of hmm's and number of gmm's

    steps/train_sat.sh --cmd "$mfcc_cmd" \
              4200 40000 $data/$train $data/lang_test $exp/tri3_ali $exp/tri4

    utils/mkgraph.sh $data/lang_test $exp/tri4 $exp/tri4/graph

    for part in test_OV test_PS test_rozhlas test_wgvat; do
          steps/decode_fmllr.sh --nj $nj_dec --cmd "$decode_cmd" \
              $exp/tri4/graph $data/$part $exp/tri4/decode_$part
    done
fi

if [ $stage -le 1 ]; then
    echo "=========================== Making data cleanup  ======================"

    # The following script cleans the data and produces cleaned data
    steps/cleanup/clean_and_segment_data.sh --nj $nj --cmd "$mfcc_cmd" \
        $data/$train $data/lang $exp/tri4 $exp/tri4_cleaned $data/train_cleaned
fi

if [ $stage -le 1 ]; then
    echo "=========================== Making ctm file ==========================="
    # This creates ctm file which we need for online decoding later?
    steps/get_train_ctm.sh --use-segments false --print-silence true \
        $data/train_cleaned $data/lang $exp/tri4_cleaned_ali
fi
# Maybe put this before train sat?
if [ $stage -le 1 ]; then
    # May
    echo "===================== Getting silence probabilities ==================="
    steps/get_prons.sh --cmd "$mfcc_cmd" \
        $data/train_cleaned $data/lang $exp/tri4_cleaned

    utils/dict_dir_add_pronprobs.sh --max-normalize true \
        $data/local/dict \
        $exp/tri4_cleaned/pron_counts_nowb.txt $exp/tri4_cleaned/sil_counts_nowb.txt \
        $exp/tri4_cleaned/pron_bigram_counts_nowb.txt $data/local/dict_sp

    utils/prepare_lang.sh $data/local/dict_sp "_SIL_" $data/local/lang_tmp $data/lang_sp

    utils/format_lm.sh $data/lang_sp $data/local/lm/$lm.gz $data/local/dict_sp/lexicon.txt $data/lang_sp_test

    steps/align_fmllr.sh --nj $nj --cmd "$mfcc_cmd" \
        $data/train_cleaned $data/lang_sp $exp/tri4_cleaned $exp/tri4_ali_train_sp
fi

# Watch out for stage changes in run_tdnn at top and in gen_e
# IMPORTANT NOTE TO GPU SUBMIT MAKE SURE THAT JOBS_FINAL IS EQUAL TO NUMBER IN GPU IN CMD.SH U TRAIN_CMD AND ALSO IT SHOULD BE THE SAME NUMBER AS ASKED IN QSUB
# MAKE SURE THAT THE MEMORY IS ALWAYS SET TO AT LEAST 10G (gpu_ram=10G) OR IT WILL CRASH
# SOMETIMES IT TAKES A WHILE TILL THE JOB QUEUES UP WHEN YOU REQUEST MORE GPU'S
# ALSO SOME PARTS WILL REQUIRE JUST qsub -cwd -j y -pe smp 10 AND OTHERS PROPER FULL GPU SUBMIT - MAKE SOME MORE NOTES ON THAT
# ALSO BE CAREFUL THAT THIS TAKES A LOT OF SPACE ~100G FOR BIG MODEL
# I WOULD NOT BE AFRAID TO USE MORE GPUS MAYBE EVEN LIKE 12 FOR FINAL JOBS AS WITH JUST 3 IT IS REALLY SLOW :/
# FOR train.py SUBMIT RUNSCRIPT FROM CPU (qsub -ced -j y -pe smp 3) AND IT WILL CALL THE GPU'S SO THAT YOU DO NOT OCCUPY GPU WITH RUN SCRIPT
# SOMETIMES WHEN CLUSTER IS VERY BUSY YOU JUST NEED TO WAIT :(

nnet_stage=0

if [ $stage -le 2 ]; then

    local/chain/run_tdnn.sh  --nj 10 --stage $nnet_stage --train-set train_cleaned  \
        --gmm tri4_cleaned --nnet3-affix _train_cleaned_rvb --chunk_per_minibatch "$chunk_per_minibatch" --jobs_initial $jobs_initial --jobs_final $jobs_final
fi
wait

# At this point extend the lexicon and language model
# USE AGAIN THE GET_PRONS RESCORING BECAUSE I HAVE MADE NEW LM AND LEXICON SO NEED TO ESTIMATE THE SILENCE PROBAS AGAIN?

if [ "$online" == true ]; then
    online_dir=exp/tdnn_sp_online
    steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf $data/lang_chain exp/nnet3/extractor exp/chain/tdnn1a_sp $online_dir

    { echo "# Activate down-scaling of silences in i-vector extractor (i-vector estimation more robust)"
      echo "--ivector-silence-weighting.silence-weight=0.001"
      echo "--ivector-silence-weighting.silence-phones=$(cat $data/lang_chain/phones/silence.csl)"
      echo "--ivector-silence-weighting.max-state-duration=100"
    } >>$online_dir/conf/online.conf

    # Create the recognition network (same way as usually, or with pruned LM),
    utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $online_dir $online_dir/graph
fi


















