# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys
import re

blacklist_beginning = ["AA", "Á", "BB", "CC", "ČČ", "DD", "ĎĎ","EE", "É", "FF", "GG", "HH", "II", "Í", "JJ", "KK", "LL", "MM", "NN", "OO", "Ó", "PP", "Q", \
        "RR", "ŘŘ", "SS", "ŠŠ", "TT", "ŤŤ", "UU", "ŮŮ", "ÚÚ", "VV", "W", "ZZ", "ŽŽ", "Ě", "Ý"]

blaclist_single = ["Á", "Ó", "Q", "W", "Ě", "Ý", "Í", "É"]

blacklist_middle = ["AA", "BB","AÁ", "CC", "ČČ", "DD", "ĎĎ","EE", "FF", "GG", "HH", "II", "JJ", "KK", "LL", "MM", "NN", "OO", "PP", \
        "RR", "ŘŘ", "SS", "ŠŠ", "TT", "ŤŤ", "UU", "ŮŮ", "ÚÚ", "VV","ZZ", "ŽŽ"]

for word in sys.stdin:
    word = word.strip()
    word = re.sub(r"[^a-zA-ZĚŠČŘŽÝÁÍÉŮÚŤĎŇÓěščřžýáíéúůťďňó]+", " ", word)
    # This line checks if word starts with beginning blacklist or whether its len is not 1 (meaning that some of the letters were from the
    # not allowed vocabulary so we will not use it further or whether it has blacklist in the middle)
    if word[:2] in blacklist_beginning or word[0] in blaclist_single or len(word.split()) != 1 or any(s in word for s in blacklist_middle) or word[0] == " " or len(word) <= 3:
        continue
    else:
        print(word)
