#!/bin/bash +x

set -euo pipefail

[ $# != 3 ] && echo "$0 <wav_key> <segments> <wav.scp>" && exit 1
key=$1
segments=$2
wav_scp=$3

tmpd=$(mktemp -d)

# Convert case of the key (sclite key in 'lc' -> to real key 'mixed-case'),
key=$(cut -d' ' -f1 $wav_scp | grep -i "^$key$")

# Copy the waveform,
rxfilename=$(awk -v key=$key '$1 == key { $1=""; print $0; }' <$wav_scp)
[ -z "$rxfilename" ] && echo "$0: Could not find '$key' in '$wav_scp'" && exit 1
if [ "|" == ${rxfilename:$((-1)):1} ]; then
  eval "$rxfilename cat >$tmpd/${key}.wav"
  wform=$tmpd/${key}.wav
else
  cp $rxfilename $tmpd/$(basename $rxfilename)
  wform=$tmpd/$(basename $rxfilename)
fi

# Filetr the segments,
awk -v key=$key '$2 == key { print $3, $4, "SP"; }' <$segments >$tmpd/${key}.txt

# Launch audacity,
echo "%%%"
echo "%%% HINT: IMPORT LABELS VIA MENU: File->Import->Labels"
echo "%%%"
audacity $wform 2>/dev/null


