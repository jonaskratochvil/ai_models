import sys

cond = False
for line in sys.stdin:
    if line.strip() == "ex":
        cond = True
        sys.stdout.write("START")
        continue
    if cond:
        sys.stdout.write(line)
        if line.strip().endswith("***"):
            break

sys.stdout.write("END")
