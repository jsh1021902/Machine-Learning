import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators=10, max_depth=1):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas     = []
        self.max_depth  = max_depth
        self.seed       = 1234
        
    def fit(self, X, y):
        n_samples = len(X)
        w         = np.ones(n_samples)/n_samples #initial uniform weights
         
        for itr in range(self.n_estimators):
            # create and train a decision tree classifier
            estimator = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.seed+itr)  # weak learner (do not change)
            estimator.fit(X, y, sample_weight=w) # use sample_weight to boost erroneous samples
            
            # compute weighted error
            pred = estimator.predict(X)
            weighted_error = w.dot(pred != y) / np.sum(w)


            # compute estimator weight
            alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-8))
            
            # update sample weights
            w = w * np.exp(-alpha * y * pred)
            w = w / np.sum(w)
           
            self.estimators.append(estimator)
            self.alphas.append(alpha)
    
    def predict(self, X):
        predictions = np.zeros(len(X))

        for estimator, estimator_weight in zip(self.estimators, self.alphas):
            pred = estimator.predict(X)
            predictions += estimator_weight * pred
        
        # apply sign function to get the final prediction
        ensemble_predictions = np.sign(predictions)
        
        return ensemble_predictions