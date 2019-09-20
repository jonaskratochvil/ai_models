# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import io
import re
import sys
import collections

if len(sys.argv) != 5:
    print(sys.argv[0] + "      ")
    sys.exit(2)

[basedic, extdic, outdic, outdic_ext] = sys.argv[1:5]

print("Merging dictionaries...")

words = collections.OrderedDict()
words_extended = collections.OrderedDict()

for dic in [basedic, extdic]:
    with io.open(dic, 'r+') as Fdic:
        for line in Fdic:
            arr = line.strip().replace("\t", " ").split(" ", 1) # Sometimes tabs are used
            [word, pronunciation] = arr
            #word = word.lower()
            if word not in words:
                words[word] = set([pronunciation.lower()])
                if dic == extdic:
                    words_extended[word] = set([pronunciation.lower()])

            else:
                words[word].add(pronunciation.lower())

with io.open(outdic, 'w', newline='\n') as Foutdic:
    for word in words:
        for pronunciation in words[word]:
            Foutdic.write(word + " " + pronunciation + "\n")


# Here we save only new words that are present in the new lexicon but not the old one
with io.open(outdic_ext, 'w', newline='\n') as Foutdic:
    for word in words_extended:
        for pronunciation in words_extended[word]:
            Foutdic.write(word + " " + pronunciation + "\n")

