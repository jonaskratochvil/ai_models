#!/usr/bin/env python3
import sys
import re
import numpy as np

data = sys.argv[1]
name = sys.argv[2]

N = 3  # this is an example
with open(data) as f:
    my_list = [chunk.strip().strip() for chunk in f.read().split('#' * N)]

my_list = np.array(my_list)
for i, part in enumerate(my_list):
    text_name = f"{name}{i+1:03}.txt"
    np.savetxt(text_name, np.array([part]), newline='\n', fmt='%s')
