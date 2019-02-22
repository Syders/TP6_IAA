import numpy as np

import utils
from bayes import GaussianBayes

def dataset_for_bayes(dataset:dict):
    X = []
    y = []
    i = 0
    for key in dataset.keys():
        for v in dataset[key]:
            X.append(v)
            y.append(i)
        i += 1
    return np.array(X), np.array(y), list(dataset.keys())



def main():
    learn, test = utils.build_dataset("data/data12.csv")
    b_X_test, b_y_test, b_conv_table = dataset_for_bayes(test)
    b_X_learn, b_y_learn, _ = dataset_for_bayes(learn)

    b = GaussianBayes()
    b.fit(b_X_learn, b_y_learn)

    print(b.score(b_X_test, b_y_test))
    

if __name__ == "__main__":
    main()
