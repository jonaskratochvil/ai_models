# Default config for BUT cluster,

export train_cmd="queue.pl --mem 10G --gpu 3"
export mfcc_cmd="queue.pl --mem 4G"
export segment_cmd="queue.pl --mem 8G --gpu 1"
#export train_cmd="queue.pl --mem 4G"
#export decode_cmd="run.pl --mem 5G"
export decode_cmd="queue.pl --mem 4G --gpu 1"
export mkgraph_cmd="queue.pl --mem 9G"
export cuda_cmd="queue.pl --gpu 1 --mem 4G"

#if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
#  queue_conf=$HOME/kaldi_queue_conf/default.conf # see example /homes/kazi/iveselyk/queue_conf/default.conf,
#  export feats_cmd="queue.pl --config $queue_conf --mem 2G --matylda 1 --max-jobs-run 50"
#  export train_cmd="queue.pl --config $queue_conf --mem 3G --matylda 0.25 --max-jobs-run 200"
#  export decode_cmd="queue.pl --config $queue_conf --mem 4G --matylda 0.05 --max-jobs-run 400"
#  export cuda_cmd="queue.pl --config $queue_conf --gpu 1 --mem 10G --tmp 40G"
#fi

