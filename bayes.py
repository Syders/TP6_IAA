import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from numpy.linalg import det
from typing import Union
from sklearn.metrics import confusion_matrix


class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=None, optim:bool=False) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)
        self.optim = optim
        
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        n_obs = X.shape[0]
        n_classes = self.mu.shape[0]
        n_features = self.mu.shape[1]

        # initalize the output vector
        y = np.empty(n_obs)
        pie_part=(n_features*np.log(2*np.pi))/2
        for i in range(n_obs):
            x=X[i]
            y_local = np.empty(n_classes)
            #calcul du c chapeau
            for j in range(n_classes):
                #-log(1/(pie_part*sigma_demi)) => log de la division
                divi=-(pie_part+(np.log(det(self.sigma[j])))/2)

                #exponentielle part
                inv_sigma=inv(self.sigma[j])
                #cree une matrice colonne de l'inverse en cas d'optimisation
                if(self.optim):
                    #dans le cas d'une optimisation on considère que la diagonale => creation d'une matrice colonne pour ça
                    inv_sigma=np.asarray([inv_sigma[j][k] for j in range(n_features) for k in range(n_features) if (j==k)]).reshape(1,n_features)

                x_self_t=np.asarray(x-self.mu[j]).reshape(1,n_features)
                exponent=np.dot((x_self_t/2*inv_sigma),x-self.mu[j]) if(self.optim) else np.dot(np.dot(x_self_t/2,inv_sigma),x-self.mu[j])

                #calcul du c 
                prior= 1 if (self.priors.all()==None) else self.priors[j]
                y_local[j] = (divi - exponent)+np.log(prior) 
                
            y[i]=np.argmax(y_local)
        return y
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_data = X.shape[0]
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        # initialization of parameters
        self.mu = np.zeros((n_classes, n_features))
        self.sigma =np.zeros((n_classes, n_features, n_features))
        # learning
        for i in range(n_classes):
            data_classe = [X[j] for j in range(n_data) if (i==y[j])]
            #mu = Esperance
            self.mu[i]=np.mean(data_classe)
            #sigma = covariance
            self.sigma[i]=np.cov(data_classe,rowvar=False)

        #self.affichage(X,y)
        



    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        pred=self.predict(X)
        print("Confusion matrix:")
        print(confusion_matrix(y, pred))
        return np.sum(y == pred) / len(X)
