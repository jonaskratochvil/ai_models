# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#import os
import re
import sys
import io

PATH = sys.argv[1]

with io.open(PATH, "r", encoding="utf-8") as file:
    my_str = file.read()

    new_string = re.sub(u"[^A-Za-zĚŠČŘŽÝÁÍÉŮÚŤĎŇÓ]", " ", my_str)

    cleaned_string = " ".join(new_string.split())

    new_string = cleaned_string.strip()

    if len(new_string) == 0:
        new_string = "_NOISE_"

with io.open(PATH, "w", encoding="utf-8") as file:
    file.write(new_string)

"""
     new_string = (
         line.replace("0", " ")
         .replace("1", " ")
         .replace("2", " ")
         .replace("%", " ")
         .replace("–", " ")
         .replace(":", " ")
         .replace("-", " ")
         .replace("3", " ")
         .replace("4", " ")
         .replace("5", " ")
         .replace("6", " ")
         .replace("7", " ")
         .replace("8", " ")
         .replace("9", " ")
         .replace("(", " ")
         .replace(")", " ")
         .replace("_", " ")
         .replace("…", " ")
         .replace("/", " ")
         .replace("\\", " ")
         .replace("!", " ")
         .replace("\"", ' ')
         .replace(".", ' ')
         .replace("+", ' ')
         .replace(";",' ')
         .replace("…", ' ')
     )
 """

