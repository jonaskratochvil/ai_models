#!/bin/bash

# TEST HOW ACCURATE THE TRANSCRIPTS ARE - DO TRAINING AND DISCART THE NOT ACCURATE ONES
# for line in segments do
    #< /dev/null ffmpeg -ss $start_time -t $duration -i $source_file -acodec copy $dest_file

# here source file is the second column from segments and dest_file is first column, start_time third column and $duration segment duration

# than cat second column of text to dest_file.trn and move both of them to some directory, moreover I can extend the silence immediately?
