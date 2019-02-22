
def build_dataset(path):
    with open(path, "r") as f:
        data = dict()
        for line in f:
            line = line.strip()
            l = line.split(",")
            if not l[-1] in data.keys():
                data[l[-1]] = []
            else:
                data[l[-1]].append(tuple(map(float, l[:-1])))
        learn = dict()
        test = dict()
        for key in data.keys():
            cut = int(len(data[key])*0.8)
            learn[key] = data[key][:cut]
            test[key] = data[key][cut:]
        return learn, test



if __name__ == "__main__":
    l,t = build_dataset("data/data2.csv")
    print(l)
    print(t)
