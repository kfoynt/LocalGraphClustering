def read_seed(filename):
    f = open(filename)
    data = f.read()
    data = data.split()
    nseedids = int(data[0])
    seedids = []
    for i in range(nseedids):
        seedids += [data[i + 1]]
    f.close()
    return (seedids,nseedids)
