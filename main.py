import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix

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
    folder='./data/'
    files=os.listdir(folder)
    for name in files:
        print("\n\nFilename=",name)
        filename=folder+name
        learn, test = utils.build_dataset(filename)
        X_test, y_test, conv_table = format_dataset(test)
        X_learn, y_learn, _ = format_dataset(learn)

        #print(X_test)
        #print("\n")
        #print(y_test)

        #Gaussian Bayes

        start = time.perf_counter()
        b = GaussianBayes()
        b.fit(X_learn, y_learn)
        pred = b.predict(X_test)

        end = time.perf_counter()
        print("\n-Gaussian Bayes:\nTime : ",(end-start))
        print("Confusion Matrix :\n",confusion_matrix(y_test, pred),"\nScore : ",score(pred, y_test))


        #K Neighbors Regressor
        success = []
        bestPredN=[]
        bestTime=0
        bestScore=0
        bestK=0

        #Test in different K
        for i in range(1,40):
            start = time.perf_counter()
            neigh = KNeighborsRegressor(n_neighbors=i,weights='uniform')
            neigh.fit(X_learn, y_learn) 
            predN = neigh.predict(X_test).astype(int)
            end = time.perf_counter()
            success.append(score(predN, y_test))
            if(bestScore<score(predN, y_test)):
                bestPredN=predN
                bestTime=end-start
                bestScore=score(predN, y_test)
                bestK=i

        print("\n-The best: K=",bestK," Neighbors Regressor:\nTime : ",bestTime)
        print("Confusion Matrix :\n",confusion_matrix(y_test, bestPredN),"\nScore : ",bestScore)

        #Affichage comparaison K Neighbors Regressor
        plt.figure(figsize=(12,6))
        plt.plot([score(pred,y_test) for x in range(40)],color='blue')
        plt.plot(range(1,40), success, color='green', linestyle='dashed', marker='o',
            markerfacecolor='blue',markersize=10)
        plt.title('Success Rate K Value')
        plt.xlabel('K value')
        plt.ylabel('Success Rate')
        plt.show()

    

if __name__ == "__main__":
    main()
