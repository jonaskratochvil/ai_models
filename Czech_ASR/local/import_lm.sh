#!/bin/bash

set -eu

[ $# != 1 ] && echo "Usage: $0 <speechdat-dir>" && exit 1

ARPA_LM_GZ=$1

lmdir=data/local/lm; mkdir -p $lmdir
uconv -f utf8 -t utf8 -x "Any-Lower" <(gunzip -c $ARPA_LM_GZ) | gzip -c >$lmdir/lm.gz

utils/format_lm_sri.sh data/lang $lmdir/lm.gz data/lang_test

