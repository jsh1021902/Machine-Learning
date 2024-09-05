import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def kernel_RBF(X, Z, gamma=0.1):
    '''
        X: numpy array [num_sample1, num_feature]
        Z: numpy array [num_sample2, num_feature]

        output
            exp( - gamma * || x - z ||^2 )
    '''
    return np.exp( - gamma*euclidean_distances(X, Z, squared=True))  # built-in pairwise distance is used for optimal computation time

def logistic(z):
    return  1./(1+np.exp(-z))




### Kernelized Logistic Regression with Ridge penalty (using Cross-Entropy Loss)
class LogisticRegression_Kernelized:
    def __init__(self, learning_rate: float = 0.05, gamma: float = 0.1, lmbda: float = 1e-2, iteration: int = 1000):
        self.learning_rate = learning_rate
        self.iteration     = iteration
        self.gamma         = gamma
        self.lmbda         = lmbda

    def fit(self, X, y):
        self.alpha   = np.zeros([len(X),1])   #initilize alpha with zero's
        self.loss    = []

        self.X       = X
        K            = kernel_RBF(self.X, self.X, self.gamma) #compute kernel matrix with hyper-parameter gamma

        ### gradient descent algorithm on alpha
        for _ in range(self.iteration):
            # Compute gradient wrt alpha (not that ridge penalty is used with the corresponding coefficient lmbda)
            gradient = np.sum(y * K.T * (1 - logistic(-y * np.matmul(K, self.alpha))), axis=1, keepdims=True) + 2 * self.lmbda * np.matmul(K,self.alpha)
            self.alpha -= self.learning_rate * gradient

            # compute loss with updated self.w
            tmp         = np.exp(- y * np.matmul(K, self.alpha))
            self.loss.append(np.mean(np.log(1 + tmp + 1e-8)))  #this is to track loss behavior


    def predict(self, Z):
        '''
            Input
                - Z: np.array([num_sample, num_feature])
            Ouput
                - np.array([num_sample, 1])
                - each element is the probability of y=1 given Z
        '''

        Kz   = kernel_RBF(self.X, Z)

        # Compute predicted probabilities using kernelized logistic regression
        pred = logistic(np.matmul(-Kz.T, self.alpha))

        return pred