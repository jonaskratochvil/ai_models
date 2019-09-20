#!/bin/bash/ python3

import sys

for line in sys.stdin:
    if len(line.strip()) == 0:
        continue
    if line.strip().endswith(":"):
        continue
    line_upper = line.strip().upper()
    line_naked = line_upper.replace(",", "").replace("%", "PROCENT").replace(
        "...", ".").replace("–", "").replace("-", "").replace("‚", "").replace("‘", "")
    print("{}".format(line_naked))
