#!/bin/bash
stage=4
data=/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/kaldi/cz_experiments/

if [ $stage -le 4 ]; then
    { 	KALDI=/home/machacek/work/elitr/kaldi-integration/kaldi-vesely-produce-time/
        [ ! -e steps ] && ln -s $KALDI/egs/wsj/s5/steps;
        [ ! -e utils ] && ln -s $KALDI/egs/wsj/s5/utils;
        #cp $KALDI/egs/wsj/s5/path.sh .;
        sed -i "s:KALDI_ROOT=.*:KALDI_ROOT=$KALDI:" path.sh;
    }
fi

if [ $stage -le 3 ]; then
    . ./path.sh

    online_dir=$data/exp/tdnn_online
    steps/online/nnet3/prepare_online_decoding.sh \
      --mfcc-config conf/mfcc_hires.conf $data/data/lang_chain exp/nnet3/extractor $data/exp/chain/tdnn1a_sp_online $online_dir

    { echo "# Activate down-scaling of silences in i-vector extractor (i-vector estimation more robust)"
      echo "--ivector-silence-weighting.silence-weight=0.001"
      echo "--ivector-silence-weighting.silence-phones=$(cat $data/data/lang_chain/phones/silence.csl)"
      echo "--ivector-silence-weighting.max-state-duration=100"
    } >>$online_dir/conf/online.conf
    # Create the recognition network (same way as usually, or with pruned LM),
    utils/mkgraph.sh --self-loop-scale 1.0 $data/data/lang_test $online_dir $online_dir/graph_pp
fi

if [ $stage -le 3 ]; then
    . path.sh
    online_dir=$data/exp/tdnn_online
    graph=$online_dir/graph_pp
    echo "BOO BOO" >utt2spk
    echo "BOO $PWD/test_audios/snemovna_with_gold_text.wav" >wav.scp
    #echo "BOO $PWD/test_audios/Europarl_test_long_fnl.wav" >wav.scp
    online2-wav-nnet3-latgen-faster \
      --online=true --do-endpointing=false --frame-subsampling-factor=3 \
      --config=$online_dir/conf/online.conf \
      --max-active=7000 --beam=15.0 --lattice-beam=6.0 --acoustic-scale=1.0 \
      --word-symbol-table=$graph/words.txt \
      $online_dir/final.mdl \
      $graph/HCLG.fst \
      ark:utt2spk \
      scp:wav.scp \
      "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >sample_swbd_20s_segment.lat.gz"
fi
rm utt2spk wav.scp
# default beam 15 lattice beam 6
# --produce-time puts time marks on output
if [ $stage -le 4 ]; then
    #. decode_path/path.sh
    . ./path_dominik.sh
    # old model
    #    online_dir=/home/jkratochvil/personal_work_ms/00/kaldi/egs/cz-speechdat-asr-example/exp/tdnn_sp_online
    #    graph=$online_dir/graph
    online_dir=$data/exp/tdnn_online
    graph=$online_dir/graph_pp
    online2-tcp-nnet3-decode-faster  \
      --verbose=2 --produce-time=true --frame-subsampling-factor=3 \
      --beam=15.0 --lattice-beam=6.0 --acoustic-scale=1.0 \
      --max-active=7000 --frames-per-chunk=20 \
      --config=$online_dir/conf/online.conf \
      --read-timeout=-1 \
      --samp-freq=16000 \
      $online_dir/final.mdl $graph/HCLG.fst $graph/words.txt
fi

