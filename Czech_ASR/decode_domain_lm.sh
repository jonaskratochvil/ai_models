#!/bin/bash
. cmd.sh
#. path.sh
data=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments

test_sets="test_OV test_PS test_rozhlas test_wgvat"
# Set the paths of our input files into variables
model=$data/exp/chain/tdnn1a_sp_online
phones_src=$data/exp/chain/tdnn1a_sp_online/phones.txt
dict_src=$data/new/local/dict
lm_src=$data/new/local/lang/

lang=$data/new/lang
dict=$data/new/dict
dict_tmp=$data/new/dict_tmp
graph=$data/new/graph
dir=$data/exp/chain/tdnn1a_sp
stage=21
nspk=6
chunk_width=140,100,60
if [ $stage -le 20 ]; then

# Prepares lexicon make sure that you have all extra_questions.txt, nonsilence_phones.txt, optional_silence.txt, silence_phones.txt copied from data/local/dict in dict_src. Also make sure that there is your new lexicon.txt - check entries such as _SIL_ _NOI_ etc. if their phoneme transcription is uppercase in both lexiconp.txt and lexicon.txt

utils/prepare_lang.sh --phone-symbol-table $phones_src $dict_src "_SIL_" $dict_tmp $dict

# Here format language model egainst the new lexicon, make sure that the best lm as obtained by local/prepare_lm.sh is in lm_src directory
gzip  $lm_src.gz
utils/format_lm.sh $dict $lm_src/lm.0.9.gz $dict_src/lexicon.txt $lang

# Here I recompile the graph - if will reside in new directory

utils/mkgraph.sh --self-loop-scale 1.0 $lang $model $graph || exit 1;

# for this to work I need to copy from /new/graph/phonemes/silence.csl to dict/phonemes and to dict/ phonemes.txt fro /new/graph/
# It updates the configuration files so that I can decode with the new model

steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf $dict exp/nnet3/extractor $data/exp/chain/tdnn1a_sp $data/new
fi

if [ $stage -le 18 ]; then
frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
rm $dir/.error 2>/dev/null || true


# This decodes whole dev directory
for data in $test_sets; do
 (
   data_affix=$(echo $data | sed s/test_//)
   # Jonas making change here from 40 to 2
   steps/nnet3/decode.sh \
     --acwt 1.0 --post-decode-acwt 10.0 \
     --extra-left-context 0 --extra-right-context 0 \
     --extra-left-context-initial 0 \
     --extra-right-context-final 0 \
     --frames-per-chunk $frames_per_chunk \
     --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
     --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
     $graph /lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/data/${data}_hires ${dir}/decode_extended_${data_affix} || exit 1
 ) || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


# This does one time decoding for certain recording
if [ $stage -le 20 ]; then
     . path.sh
     online_dir=$data/exp/tdnn_online
     echo "BOO BOO" >utt2spk
     #echo "BOO $PWD/test_audios/Europarl_test_long_fnl.wav" >wav.scp
     echo "BOO $PWD/test_audios/WG_test_67.wav" >wav.scp
     online2-wav-nnet3-latgen-faster \
       --online=false --do-endpointing=false --frame-subsampling-factor=3 \
       --config=$data/new/conf/online.conf \
       --max-active=7000 --beam=15.0 --lattice-beam=6.0 --acoustic-scale=1.0 \
       --word-symbol-table=$graph/words.txt \
       $online_dir/final.mdl \
       $graph/HCLG.fst \
       ark:utt2spk \
       scp:wav.scp \
       "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >sample_swbd_20s_segment.lat.gz"
fi

######################################################
# This does decoding with old model before Brno
if [ $stage -le 20 ]; then
     . path.sh
     online_dir=/home/jkratochvil/personal_work_ms/kaldi/egs/cz-speechdat-asr-example/exp/tdnn_sp_online
     graph=$online_dir/graph
     echo "BOO BOO" >utt2spk
     echo "BOO $PWD/test_audios/snemovna_with_gold_text.wav" >wav.scp
     online2-wav-nnet3-latgen-faster \
       --online=true --do-endpointing=false --frame-subsampling-factor=3 \
       --config=$data/new/conf/online.conf \
       --max-active=7000 --beam=15.0 --lattice-beam=6.0 --acoustic-scale=1.0 \
       --word-symbol-table=$graph/words.txt \
       $online_dir/final.mdl \
       $graph/HCLG.fst \
       ark:utt2spk \
       scp:wav.scp \
       "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >sample_swbd_20s_segment.lat.gz"
fi
#######################################################

# This starts TCP server - do it from sol1!

if [ $stage -le 20 ]; then
     {   KALDI=/home/machacek/work/elitr/kaldi-integration/kaldi-vesely-produce-time/
         [ ! -e steps ] && ln -s $KALDI/egs/wsj/s5/steps;
         [ ! -e utils ] && ln -s $KALDI/egs/wsj/s5/utils;
         #cp $KALDI/egs/wsj/s5/path.sh .;
         sed -i "s:KALDI_ROOT=.*:KALDI_ROOT=$KALDI:" path.sh;
     }
fi

if [ $stage -le 21 ]; then
     #. decode_path/path.sh
     . ./path_dominik.sh
     online_dir=$data/exp/tdnn_online
     online2-tcp-nnet3-decode-faster  \
       --verbose=2 --produce-time=true --frame-subsampling-factor=3 \
       --beam=15.0 --lattice-beam=6.0 --acoustic-scale=1.0 \
       --max-active=7000 --frames-per-chunk=20 \
       --config=$data/new/conf/online.conf \
       --read-timeout=-1 \
       --samp-freq=16000 \
     $online_dir/final.mdl $graph/HCLG.fst $graph/words.txt
fi

