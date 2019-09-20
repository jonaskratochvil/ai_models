# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import sys
session = sys.argv[1]

for line in sys.stdin:
    if "schůze, část č." not in line:
        continue
    line = line.strip()
    first_part = line.split(",")[0]
    first_time = first_part.split()[0].replace(":", "")
    second_time = first_part.split()[2].replace(":", "")
    if len(first_time) < 4:
        first_time = "0"+first_time
    if len(second_time) < 4:
        second_time = "0"+second_time
    number = line.split("č.")[1].split()[0]
    meeting = line.split("schůze,")[0].split()[-1].replace(".", "")
    if int(meeting) != int(session):
        continue
    final_time = str(first_time) + str(second_time) + " " + number
    print(final_time)
