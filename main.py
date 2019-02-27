import time
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

#from https://scikit-learn.org
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def main():
    folder='./data/'
    files=os.listdir(folder)
    for name in files:
        print("\n\n-Filename=",name)
        filename=folder+name
        rates=[0.8,0.5,0.2]
        for rate in rates:
        	print("\n\n\n-Actual rate:",rate)
	        learn, test = utils.build_dataset(filename, random=False,rate=rate)
	        X_test, y_test, labels = format_dataset(test)
	        X_learn, y_learn, _ = format_dataset(learn)
	        data_dim = len(X_test[0])

	        #Gaussian Bayes

	        start = time.perf_counter()
	        b = GaussianBayes()
	        b.fit(X_learn, y_learn)
	        pred = b.predict(X_test)

	        end = time.perf_counter()
	        print("\n-Gaussian Bayes:\nTime : ",(end-start))
	        print("Confusion Matrix :\n",confusion_matrix(y_test, pred),"\nScore : ",score(pred, y_test))
	        plot_confusion_matrix(confusion_matrix(y_test, pred), labels, title="Confusion matrix, Bayes, dim=%d"%data_dim)
	        


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
	        plot_confusion_matrix(confusion_matrix(y_test, bestPredN), labels, title="Confusion matrix, KNN, k=%d, dim=%d"%(bestK, data_dim))

	        #Affichage comparaison K Neighbors Regressor
	        plt.figure(figsize=(12,6))
	        plt.plot([score(pred,y_test) for x in range(40)],color='blue', label="Bayes")
	        plt.plot(range(1,40), success, color='green', linestyle='dashed', marker='o',
	            markerfacecolor='green',markersize=10, label="KNN")
	        plt.title('Success Rate (higher is better), dim=%d'%data_dim)
	        plt.xlabel('K value')
	        plt.ylabel('Success Rate')
	        plt.legend()
	        plt.show()

    

if __name__ == "__main__":
    main()
