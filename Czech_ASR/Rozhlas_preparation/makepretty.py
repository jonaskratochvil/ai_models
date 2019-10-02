#!/bin/bash/ python3

import sys

for line in sys.stdin:
    if len(line) == 0:
        continue
    line_upper = line.upper()
    line_naked = line_upper.replace(",", "").replace("%", "PROCENT").replace(
        "...", ".").replace("–", "").replace("‚", "").replace("‘", "").replace("PS", "POSLANECKÁ SNĚMOVNA").replace("EU", "EVROPSKÁ UNIE")
    print("{}".format(line_naked))
