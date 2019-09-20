#!/usr/bin/env python2
# Copyright 2014  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import sys
import codecs
import re
if len(sys.argv) != 7:
  print __file__,'<segs-in>','<text-in>','<segs-trimmer>','<segs-out>','<text-out>','<utt2new_utt>','\n'
  print sys.argv
  sys.exit(0)

segs_in_file, text_in_file, segs_trimmer_file, segs_out_file, text_out_file, utt2new_utt_file = sys.argv[1:]

# Read the inputs,
segs_in = np.loadtxt(segs_in_file, dtype='object')
segs_trimmer = np.loadtxt(segs_trimmer_file, dtype='object')
with codecs.open(text_in_file, 'r', 'utf-8') as f:
  text_in = dict([(l[:re.search(r'[ \t]',l).start()], l[re.search(r'[ \t]',l).start():].strip()) for l in f ])

# Convert segs trimmer to per-frame representation,
trimmer = dict()
for spk in np.unique(segs_trimmer[:,1]):
  # Get begin, end times in frames,
  beg = (segs_trimmer[segs_trimmer[:,1] == spk][:,2].astype(float)*100).astype(int)
  end = (segs_trimmer[segs_trimmer[:,1] == spk][:,3].astype(float)*100).astype(int)
  # Get per-frame representation,
  x = np.zeros(end[-1]+3000) # 30s more
  for b,e in zip(beg,end): x[b:e] = 1
  trimmer[spk] = x

# Trim the segments,
segs_out=[]
text_out=[]
utt2new_utt=[]
for (utt,spk,beg,end) in segs_in:
  b = int(float(beg)*100.0)
  e = int(float(end)*100.0)
  try:
    x = trimmer[spk][b:e]
    if np.sum(x) == 0: raise NoSpeechFound
    b2 = b + np.argmax(x) # argmax returns 1st speech frame,
    e2 = b + np.argmax(np.cumsum(x)) # argmax returns last speech frame,
  except:
    # x is missing or had no speech, keep original segment boundaries...
    # The segments with completely wrong annotation are excluded when aligning.
    # (tried to remove empty segments on Bengali -> hit of 0.4% WER with tri5b)
    b2 = b
    e2 = e
  if e2-b2 < 25: continue # set minimum segment-length to 25 frames,
  new_utt = '%s_%07d_%07d' % (spk,b2,e2)
  segs_out.append('%s %s %06.2f %06.2f' % (new_utt,spk,b2/100.0,e2/100.0))
  utt2new_utt.append('%s %s' % (utt, new_utt))
  try:
    text_out.append('%s %s' % (new_utt,text_in[utt]))
  except:
    pass

# Save the outputs,
np.savetxt(segs_out_file, segs_out, fmt='%s', delimiter='\n')
with codecs.open(text_out_file, 'w', 'utf-8') as f : f.write('\n'.join(text_out))
np.savetxt(utt2new_utt_file, utt2new_utt, fmt='%s', delimiter='\n')

# Done!
