#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License. #
#
# Makes train/test splits
# local/voxforge_data_prep.sh --nspk_test ${nspk_test} ${SELECTED} || exit 1
# create files: (TYPE=train|test)
#   a) ${TYPE}_trans.txt: ID transcription capitalized! No interputction
#   b) ${TYPE}_wav.scp: ID path2ID.wav
#   c) $TYPE.utt2spk: ID-recording ID-speaker
#   s) $TYPE.spk2utt
#   e) $TYPE.spk2gender  all speakers are male
# we have ID-recording = ID-speaker

# The vystadial data are specific by having following marks in transcriptions
# _INHALE_
# _LAUGH_
# _EHM_HMM_
# _NOISE_
# _EHM_HMM_
# _SIL_

# renice 20 $$

every_n=1

[ -f path.sh ] && . ./path.sh # source the path.
. utils/parse_options.sh || exit 1;


if [ $# -ne 4 ] ; then
    echo "Usage: local/data_split.sh [--every-n 30] <data-directory>  <local-directory> <LMs> <Test-Sets> <tgt-dir>";
    exit 1;
fi

DATA=$1; shift
locdata=$1; shift
LMs=$1; shift
test_sets=$1; shift
tgt_dir=$1; shift

echo "LMs $LMs  test_sets $test_sets"


echo "=== Starting initial Rozhlas data preparation ..."
echo "--- Making test/train data split from $DATA taking every $every_n recording..."

mkdir -p $locdata

i=0
for s in $test_sets train ; do
    mkdir -p $locdata/$s
    for pardir in /lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/new_data/train/experiment_dir/$s/* ; do
	    for childir in $pardir/* ; do
   	 	ls $childir | sed -n /.*wav$/p |\
   	 	    while read wav ; do
		        ((i++)) # bash specific
		        if [[ $i -ge $every_n ]] ; then
		            i=0
                    # Put condition to skip corrupted audio files so there is no problem with durations later
                    if [[ $(stat --printf="%s" $childir/$wav) -ge 100 ]] ; then
           	            sxcmd="-c 1 -r 16000"

                        if [[ -f $wav.spk.trn ]] ; then
                            rm $wav.spk.trn
                        fi
            		    pwav=$childir/$wav
                        # convert numeric values to text
                        #cat $childir/$wav.trn | PYTHONIOENCODING=utf-8 python3 \
                        #    local/numtotext.py > $childir/$wav.tmp.trn
                        #rm $childir/$wav.trn
                        #mv $childir/$wav.tmp.trn $childir/$wav.trn
                        python3 local/numtotext.py $childir/$wav.trn
                        # get rid of all non-alphabet symbols
                        python3 local/stripstring.py $childir/$wav.trn
                        # changel rate and channel of each audio
                        #sox -r 16000 -c 1 $childir/$wav $childir/$wav.wav
                        #mv $childir/$wav $childir/$wav.wav
                        #rm $childir/$wav

                        #mv $childir/$wav.wav $childir/$wav
                        # This takes the base name

                        filename="${wav%_*}"
                        trn=`cat $childir/$wav.trn`
           	    	    echo "$wav $pwav" >> $locdata/$s/wav.scp
           	    	    echo "$wav $filename" >> $locdata/$s/utt2spk
           	    	    echo "$filename $wav" >> $locdata/$s/spk2utt
           	    	    echo "$wav $trn" >> $locdata/$s/trans.txt
           	    	    # Ignoring gender -> label all recordings as male
           	    	    echo "$wav M" >> $locdata/spk2gender
                        echo "$trn" >> $locdata/corpus.txt
                    else
                        echo "File $childir/$wav is corrupted and will not be used for training."
                        continue
                    fi
        	    fi
    		done # while read wav
	    done # childir
    done # for pardir

    for f in wav.scp utt2spk spk2utt trans.txt ; do
       sort "$locdata/$s/$f" -k1 -u -o "$locdata/$s/$f"  # sort in place
    done # for f

done # for in $test_sets train

echo "Set 1:1 relation for spk2utt: spk in $test_sets AND train, sort in place"
sort "$locdata/spk2gender" -k1 -o "$locdata/spk2gender"

echo "--- Distributing the file lists to train and ($test_sets x $LMs) directories ..."
mkdir -p $WORK/train
cp $locdata/train/wav.scp $WORK/train/wav.scp || exit 1;
cp $locdata/train/trans.txt $WORK/train/text || exit 1;
cp $locdata/train/spk2utt $WORK/train/spk2utt || exit 1;
cp $locdata/train/utt2spk $WORK/train/utt2spk || exit 1;
cp $locdata/corpus.txt $WORK/local
utils/filter_scp.pl $WORK/train/spk2utt $locdata/spk2gender > $WORK/train/spk2gender || exit 1;

for s in $test_sets ; do
    for lm in $LMs; do
        tgt_dir=$WORK/${s}_`basename ${lm}`
        mkdir -p $tgt_dir
        cp $locdata/${s}/wav.scp $tgt_dir/wav.scp || exit 1;
        cp $locdata/${s}/trans.txt $tgt_dir/text || exit 1;
        cp $locdata/${s}/spk2utt $tgt_dir/spk2utt || exit 1;
        cp $locdata/${s}/utt2spk $tgt_dir/utt2spk || exit 1;
        utils/filter_scp.pl $tgt_dir/spk2utt $locdata/spk2gender > $tgt_dir/spk2gender || exit 1;
    done
done
