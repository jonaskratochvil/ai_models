#!/usr/bin/env python3
import sys

buf = b''
ind = 0
while True:
    #ind = 0
    r = sys.stdin.buffer.read(1)
    buf += r
    if r in (b'\r', b'\n'):
        line = buf.decode("utf-8")
        x = line.index("}")+1
        line = line[x:].strip()
        if line:
            print(line[ind:].strip())
            ind = len(line)
            sys.stdout.flush()
        buf = b''
