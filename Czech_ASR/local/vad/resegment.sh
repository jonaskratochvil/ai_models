#!/bin/bash +x
# Copyright 2014  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

# Begin configuration.
trim_old_segments=false # false : import new segmentation; true : trim silence-margins in 'srcdata/segments'
segmenter_opts="--smooth-window 31 --speech-threshold -0.5 --extension-length 30"
stage=0
jpm=5
cmd='run.pl'
# End configuration.

echo "$0 $@"  # Print the command line for logging
echo

[ -f path.sh ] && . ./path.sh;

. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
   echo "usage: $0 [options] vad_script data data_segmented"
   echo "example: $0 apply_vad.sh data/dev data/dev-segmented"
   echo ""
   echo "options:"
   echo "  --cmd 'queue.pl'"
   echo ""
   exit 1;
fi

set -euo pipefail

vad_script=$1
srcdata=$2
data=$3

# Absolute paths,
[ ${vad_script:0:1} != '/' ] && vad_script=$PWD/$vad_script
[ ${srcdata:0:1} != '/' ] && srcdata=$PWD/$srcdata
[ ${data:0:1} != '/' ] && data=$PWD/$data

mkdir -p $data/vad/{seg,png,log,batch}

[ ! -f $srcdata/wav.scp ] && echo "Missing $srcdata/wav.scp" && exit 1

# Run the vad,
if [ $stage -le 0 ]; then
  # generate batch,
  awk -v data=$data -v vad_script="$vad_script --segmenter-opts '$segmenter_opts -g $data/vad/png'" \
    '{ # Parse the  "wav.scp" linet : <key> <wav-rspec>
       key=$1; $1=""; wav_rspec=gensub(/^\s+/,"","",$0);

       # In key, replace the slashes: "/" -> "^",
       key_noslash=gensub(/\//,"^","g",key)

       # Output file with segmentation,
       seg=sprintf("%s/vad/seg/%s.seg", data, key_noslash);

       # Print the command for the BATCH file,
       printf("[ -f %s ] && echo \"%s already exists\" && exit 0 || %s \"%s\" %s\n", seg, seg, vad_script, wav_rspec, seg);
     }' <$srcdata/wav.scp >$data/vad/BATCH

  # split the batch,
  n_wav=$(cat $srcdata/wav.scp | wc -l)
  nj=$((1+n_wav/jpm))
  scps=""; for (( ii=1; ii<=nj; ii++ )); do scps="$scps $data/vad/batch/$ii"; done
  utils/split_scp.pl $data/vad/BATCH $scps

  # run the task,
  $cmd JOB=1:$nj $data/vad/log/vad.JOB.log \
    bash $data/vad/batch/JOB
fi

# Prepare output data dir,
if [ $trim_old_segments == false ]; then # no previous segs,
  [ $srcdata != $data ] && cp $srcdata/* $data # copy all files,
  find $data/vad/seg/ -name "*.seg" | xargs -I{} cat {} | tr '^' '/' | grep -v '^[ ]*$' | sort > $data/segments # key-slashes '^' -> '/',
  awk '{ print $1, $2; }' $data/segments > $data/utt2spk # key from 'wav.scp' is a 'spk',
  utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
else # re-trim the previous segments, leave-out silence boundaries,
  cp $srcdata/segments $data/segments-old
  find $data/vad/seg/ -name "*.seg" | xargs -I{} cat {} | tr '^' '/' | grep -v '^[ ]*$' | sort > $data/segments-vad # key-slashes '^' -> '/',
  local/vad/trim_old_segments.py $data/segments-old $srcdata/text $data/segments-vad $data/segments $data/text $data/utt2new_utt
  utils/apply_map.pl -f 1 --permissive $data/utt2new_utt <$srcdata/utt2spk >$data/utt2spk # port the 'spk' labels,
  utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
  cp $srcdata/wav.scp $data
fi

# Fix the dir,
utils/fix_data_dir.sh $data

echo "$0: success"
