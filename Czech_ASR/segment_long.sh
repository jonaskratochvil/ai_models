#!/bin/bash

# The current setup runs the extended lexicon and LM versions where the lexicon was updated for the words in transcriptions. Directories needed to be cleaner if new experiment is to be run are : data_to_segment data_segmented train_tdnn1_cleane

stage=2
data=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/data
#exp=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/exp
exp=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/
. ./path.sh
. ./cmd.sh
# This requires text file (utt.wav text on one line), wav.scp, spk2utt, utt2spk, text2utt - vsechny tyto jen utt.wav utt.wav
# For all nexxessary data preparation I have script in MASNEMOVNA which should prepare all the above
if [ $stage -le 2 ]; then
    steps/make_mfcc.sh --cmd "$mfcc_cmd" --nj 4 $data/data_to_segment $exp/exp/make_mfcc_to_segment mfccs_to_segment
    steps/compute_cmvn_stats.sh $data/data_to_segment $exp/exp/make_mfcc_to_segment mfccs_to_segment
fi
# This needs GPU both of them below, ssh to kronos, make sure that cmd.sh has --gpu x and submit through qsub -cwd -j y -pe smp x
if [ $stage -le 2 ]; then
    #steps/cleanup/segment_long_utterances_nnet3.sh --nj 4 --cmd "$segment_cmd" --extractor exp/nnet3/extractor $exp/chain/tdnn1a_sp $data/lang_chain \
    #    $data/data_to_segment $data/data_to_segment/text $data/data_to_segment/utt2text $data/data_segmented/ $exp/data_segmented
    steps/cleanup/segment_long_utterances_nnet3.sh --nj 4 --cmd "$segment_cmd" --extractor exp/nnet3/extractor $exp/new $exp/new/lang \
        $data/data_to_segment $data/data_to_segment/text $data/data_to_segment/utt2text $data/data_segmented/ $exp/exp/data_segmented
fi

utils/fix_data_dir.sh $data/data_segmented

if [ $stage -le 3 ]; then
    #steps/cleanup/clean_and_segment_data_nnet3.sh  --nj 4 --cmd "$segment_cmd" --extractor exp/nnet3/extractor $data/data_segmented/ $data/lang_chain $exp/chain/tdnn1a_sp $exp/tdnn1a_cleanup $data/train_tdnn1_cleane
    steps/cleanup/clean_and_segment_data_nnet3.sh  --nj 4 --cmd "$segment_cmd" --extractor exp/nnet3/extractor $data/data_segmented/ $exp/new/lang $exp/new $exp/exp/tdnn1a_cleanup $data/train_tdnn1_cleane
fi
