# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import io
import re
import sys
import collections

if len(sys.argv) != 4:
    print(sys.argv[0] + "      ")
    sys.exit(2)

[basedic, extdic, outdic] = sys.argv[1:4]

print("Merging dictionaries...")

words = collections.OrderedDict()
for dic in [basedic, extdic]:
    with io.open(dic, 'r+') as Fdic:
        for line in Fdic:
            arr = line.strip().replace("\t", " ").split(" ", 1) # Sometimes tabs are used
            [word, pronunciation] = arr
            word = word.lower()
            if word not in words:
                words[word] = set([pronunciation.lower()])
            else:
                words[word].add(pronunciation.lower())

with io.open(outdic, 'w', newline='\n') as Foutdic:
    for word in words:
        for pronunciation in words[word]:
            Foutdic.write(word + " " + pronunciation + "\n")
