import numpy as np

from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode


class BaggingClassifier:
    def __init__(self, n_estimators=100, max_depth=None): 
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.estimators   = []
        self.seed         = 1234
    
    def fit(self, X, y):
        n_samples = X.shape[0]

        for itr in range(self.n_estimators):

            # bootstrap sample with size n_samples
            # you may use np.random.choice with "replacement=True" to implement bootstrapping

            np.random.seed(self.seed+itr)
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_samples, y_samples = X[sample_indices], y[sample_indices]

            # create and train DecisionTreeClassifier boostrapped samples (use self.max_depth for max_depth)
            estimator = DecisionTreeClassifier(max_depth=self.max_depth)
            estimator.fit(X_samples, y_samples)


            # add the trained estimator to the list of estimators
            self.estimators.append(estimator)

    
    def predict(self, X):
        # Make predictions using each estimator
        predictions = np.zeros((self.n_estimators, X.shape[0]))
        for i, estimator in enumerate(self.estimators):
            predictions[i] = estimator.predict(X)
        
        # Take majority vote for each sample 
        # You may use scipy.stats.mode to avoid implementing "mode" operator from scratch
        y_pred, _ = mode(predictions, axis=0)
        y_pred = y_pred.ravel()
        return y_pred