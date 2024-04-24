for i in range(4, 17):
    fname = f"weak-seq-{i}.txt"
    print(i, open(fname, "r").read())
