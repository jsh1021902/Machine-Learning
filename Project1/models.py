import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = None
        self.conditional_prob = None
        self.m = 1 #smoothing factor
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        

        # Compute prior probabilities of each class: 'self.prior'
        self.prior = np.zeros(n_classes)
        for idx, c in enumerate(self.classes):
            self.prior[idx] = np.sum(y == c) / n_samples
        

        # Compute conditional probabilities of each feature given each class:  'self.conditional_prob
        self.conditional_prob = np.zeros((n_classes, n_features), dtype=np.ndarray)
        # use file type dict() for each element in self.conditional_prob  (this will be an easier way to find the frequency for each category of a given feature)
        for c_idx, c in enumerate(self.classes):
            samples_in_class = X[y == c]
            for f_idx in range(n_features):
                self.conditional_prob[c_idx, f_idx] = dict()
                unique_vals, counts = np.unique(samples_in_class[:, f_idx], return_counts=True)
                total_counts = np.sum(counts) + self.m * len(unique_vals)
                for u_val, count in zip(unique_vals, counts):
                    self.conditional_prob[c_idx, f_idx][u_val] = (count + self.m) / total_counts
        return self
        
    def predict(self, X):
        y_pred = np.zeros(len(X), dtype=np.int8)

        for i, sample in enumerate(X):
            probabilities = np.zeros(len(self.classes))
            for c_idx, c in enumerate(self.classes):
                prob = np.log(self.prior[c_idx])
                for f_idx, val in enumerate(sample):
                    if val in self.conditional_prob[c_idx][f_idx]:
                        prob += np.log(self.conditional_prob[c_idx][f_idx][val])
                probabilities[c_idx] = prob
            y_pred[i] = self.classes[np.argmax(probabilities)]        
            
        return y_pred
