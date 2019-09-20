#!/bin/bash

set -eu

[ $# != 1 ] && echo "Usage: $0 <speechdat-dir>" && exit 1

dir=$1

# Note:
# - we create a complete data-dir 'speechdat_cz',
# - then we split to train/test data according to speaker-lists,

# create the data-dir,
data=data/speechdat_cz; mkdir -p $data

# create mapping for 'utt' in a format 'utt -> spk_utt' (speaker is prefix in utt-key),
tail -n+2 $dir/index/contents.lst | \
  uconv -f 8859-2 -t utf8 -x "Any-Lower" | \
  awk '{ utt=$3; spk=$5; sub(/.csa$/,"",utt); print utt, spk"_"utt; }' \
  >$data/.utt_mapping

# wav.scp,
find $dir -name '*.csa' | awk '{ utt=$1;
                                 gsub(/.*\//,"",utt);
                                 sub(/.csa$/,"",utt);
                                 printf("%s sox -t raw -r8k -e a-law %s -t wav -e signed-integer - |\n", utt, $0); }' | \
                          utils/apply_map.pl -f 1 $data/.utt_mapping >$data/wav.scp

# utt2spk,
tail -n+2 $dir/index/contents.lst | awk '{ print gensub(".CSA","","1",$3), $5 }' | uconv -f 8859-2 -t utf8 -x "Any-Lower" | \
  utils/apply_map.pl -f 1 $data/.utt_mapping >$data/utt2spk
# spk2utt,
utils/utt2spk_to_spk2utt.pl <$data/utt2spk >$data/spk2utt

# text,
tail -n+2 $dir/index/contents.lst | uconv -f 8859-2 -t utf8 -x "Any-Lower" | cut -f 3,9- | awk '{ sub(/.csa$/,"",$1); print; }' | \
  utils/apply_map.pl -f 1 $data/.utt_mapping >$data/text

# make sure the data is prepared correctly,
utils/fix_data_dir.sh $data


# ------------------------------
# split the data to train/test,
#
ses_list_train=$dir/index/a3trncs.ses
ses_list_test=$dir/index/a3tstcs.ses

# convert ses to utt-lists,
awk -v ses_list=$ses_list_train 'BEGIN{ while(getline < ses_list) { gsub(/\s/,"",$1); get_those[$1]=1; }; }
                                 { ses=gensub(/^.*\\SES/,"","g",$2);
                                   if (get_those[ses] == 1) {
                                     print $5"_"gensub(/\.CSA$/,"","g",$3);
                                   }
                                 }' $dir/index/contents.lst | uconv -f 8859-2 -t utf8 -x "Any-Lower" >data/train_utts

awk -v ses_list=$ses_list_test 'BEGIN{ while(getline < ses_list) { gsub(/\s/,"",$1); get_those[$1]=1; }; }
                                { ses=gensub(/^.*\\SES/,"","g",$2);
                                  if (get_those[ses] == 1) {
                                    print $5"_"gensub(/\.CSA$/,"","g",$3);
                                  }
                                }' $dir/index/contents.lst | uconv -f 8859-2 -t utf8 -x "Any-Lower" >data/test_utts

# subset the data-dirs,
utils/subset_data_dir.sh --utt-list data/train_utts $data data/train_noreseg
utils/subset_data_dir.sh --utt-list data/test_utts $data data/test

# remove the '[...]' tokens from train,
mv data/train_noreseg/text data/train_noreseg/text.orig
sed 's:\[[a-z]*\]::g; s:  *: :g;' data/train_noreseg/text.orig >data/train_noreseg/text

# remove the '[...]' tokens from text,
mv data/test/text data/test/text.orig
sed 's:\[[a-z]*\]::g; s:  *: :g;' data/test/text.orig >data/test/text

