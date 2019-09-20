#!/bin/bash
# -*- coding: utf-8 -*-

# we have ID-recording = ID-speaker

# The vystadial data are specific by having following marks in transcriptions
# _INHALE_
# _LAUGH_
# _NOISE_
# _EHM_HMM_
# _SIL_

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
for s in $test_sets train; do
    mkdir -p $locdata/$s
    for pardir in /lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/new_data/$s/* ; do
	    for parpardir in $pardir/* ; do
            for childir in $parpardir/* ; do
            i=0
   	 	    ls $childir | sed -n /.*wav$/p |\
   	 	        while read wav ; do
		            ((i++)) # bash specific
		            if [[ $i -ge $every_n ]] ; then
		                #i=0
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
                            echo "$wav $wav" >> $locdata/$s/utt2spk
                            echo "$wav $wav" >> $locdata/$s/spk2utt
                            echo "$wav $trn" >> $locdata/$s/trans.txt
                            # Ignoring gender -> label all recordings as male
                            echo "$wav M" >> $locdata/spk2gender
                            echo "$trn" >> $locdata/corpus.txt

                            #echo "${filename}_${i} $pwav" >> $locdata/$s/wav.scp
           	    	        #echo "${filename}_${i} $filename" >> $locdata/$s/utt2spk
           	    	        #echo "$filename ${filename}_${i}" >> $locdata/$s/spk2utt
           	    	        #echo "${filename}_${i} $trn" >> $locdata/$s/trans.txt
           	    	        # Ignoring gender -> label all recordings as male
           	    	        #echo "$filemame M" >> $locdata/spk2gender
                            #echo "$trn" >> $locdata/corpus.txt
                        else
                            echo "File $childir/$wav is corrupted and will not be used for training."
                            continue
                        fi
        	        fi
                done
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
