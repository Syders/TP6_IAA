import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

import utils
from bayes import GaussianBayes

def format_dataset(dataset:dict):
    X = []
    y = []
    i = 0
    for key in dataset.keys():
        for v in dataset[key]:
            X.append(v)
            y.append(i)
        i += 1
    return np.array(X), np.array(y), list(dataset.keys())

def score(prediction:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == prediction) / len(prediction)

def main():
    learn, test = utils.build_dataset("data/data3.csv")
    X_test, y_test, conv_table = format_dataset(test)
    X_learn, y_learn, _ = format_dataset(learn)

    #print(X_test)
    #print("\n")
    #print(y_test)

    start = time.perf_counter()
    b = GaussianBayes()
    b.fit(X_learn, y_learn)
    pred = b.predict(X_test)
    end = time.perf_counter()
    print("Time : ",(end-start),"  Score : ",score(pred, y_test))

    start = time.perf_counter()
    neigh = KNeighborsRegressor(n_neighbors=5)
    neigh.fit(X_learn, y_learn) 
    pred = neigh.predict(X_test)
    end = time.perf_counter()
    print("Time : ",(end-start),"  Score : ",score(pred, y_test))

    

if __name__ == "__main__":
    main()
