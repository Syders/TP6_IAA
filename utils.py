import numpy as np


def build_dataset(path,rate, random=False):
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
            if not random :
                cut = int(len(data[key])*rate)
                learn[key] = data[key][:cut]
                test[key] = data[key][cut:]
            else :
                rateCut=1-rate
                selection_test = np.random.choice(len(data[key]), size=(int(len(data[key])*rateCut)+1), replace=False)
                selection_learn = np.delete(np.arange(len(data[key])), selection_test)
                learn[key] = np.array(data[key])[selection_learn].tolist()
                test[key] = np.array(data[key])[selection_test].tolist()
        return learn, test



if __name__ == "__main__":
    l,t = build_dataset("data/data2.csv", random=True)
    print(l)
    print(t)
