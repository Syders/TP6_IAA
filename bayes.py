import numpy as np
import math
from typing import Union


class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=None, diag:bool=False) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)
        self.diag = diag
    
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
        # log of the probability
        score = lambda x, i: -0.5*(n_features*math.log(2*math.pi) + \
            math.log(np.linalg.det(self.sigma[i])) + \
            np.dot((x-self.mu[i]).T, np.dot(np.linalg.inv(self.sigma[i]), (x-self.mu[i]))))
        
        for i in range(n_obs):
            y[i] = np.argmax([(score(X[i],c) + (0 if self.priors is None else math.log(self.priors[c])) ) for c in range(n_classes)])
                
        return y
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_features = X.shape[1]
        classes = np.unique(y)
        n_classes = len(classes)
        # initialization of parameters
        self.mu = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features, n_features))
        # learning
        for c in range(n_classes):
            self.mu[c] = np.mean([X[i] for i in range(len(X)) if y[i] == classes[c]], axis=0)
            if self.diag :
                cov = np.cov([X[i] for i in range(len(X)) if y[i] == classes[c]], rowvar=False)
                self.sigma[c] = cov*np.identity(n_features)
            else:
                self.sigma[c] = np.cov([X[i] for i in range(len(X)) if y[i] == classes[c]], rowvar=False)

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)
