l = (4, 9, 16, 25, 36)
for i in l :
    fname = f"strong-parallel-{i}.txt"
    print(i, open(fname, "r").read())
