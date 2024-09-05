import numpy as np


def logistic(X, w):
    return  1./(1+np.exp(-np.matmul(X, w)))

def loss_CE(X, y, w):
    y_hat_ = logistic(X, w)
    return np.mean( - (y * np.log(y_hat_+1e-8) + (1-y) * np.log(1.-y_hat_+1e-8)))

def loss_MSE(X, y, w):
    y_hat_ = logistic(X, w)
    return np.mean( (y - y_hat_)**2) 



### Logistic Regression with Cross-Entropy Loss
class LogisticRegression_CE:
    def __init__(self, learning_rate: float = 0.05, iteration: int = 1000, random_state: int = 1234):
        self.learning_rate = learning_rate
        self.iteration     = iteration
        self.random_state  = random_state

    def fit(self, X, y):
        X_ = np.concatenate([X, np.ones([len(X), 1])], axis=1)  #this is to incorporate bias term

        np.random.seed(self.random_state)
        self.w    = np.random.randn(np.shape(X_)[1],1)
        self.loss = []

        ### gradient descent algorithm
        for _ in range(self.iteration):

            ### GD with CE Loss
            grad = (-1 / len(X_)) * np.dot(X_.T, (y - logistic(X_, self.w)))
            self.w -= self.learning_rate * grad

            self.loss.append(loss_CE(X_, y, self.w))  #this is to track loss behavior (do not remove it)
    
    def predict(self, X):
        return logistic(np.concatenate([X, np.ones([len(X), 1])], axis=1), self.w)



### Logistic Regression with MSE Loss
class LogisticRegression_MSE:
    def __init__(self, learning_rate: float = 0.05, iteration: int = 1000, random_state: int = 1234):
        self.learning_rate = learning_rate
        self.iteration     = iteration
        self.random_state  = random_state

    def fit(self, X, y):
        X_ = np.concatenate([X, np.ones([len(X), 1])], axis=1)  #this is to incorporate bias term

        np.random.seed(self.random_state)
        self.w    = np.random.randn(np.shape(X_)[1],1)
        self.loss = []

        ### gradient descent algorithm
        for _ in range(self.iteration):

            ### GD with MSE Loss
            y_hat_ = logistic(X_, self.w)
            grad = (-2 / len(y)) * np.dot(X_.T, (y - y_hat_) * y_hat_ * (1 - y_hat_))
            self.w -= self.learning_rate * grad

            self.loss.append(loss_MSE(X_, y, self.w))  #this is to track loss behavior (do not remove it)
    
    def predict(self, X):
        return logistic(np.concatenate([X, np.ones([len(X), 1])], axis=1), self.w)




### Logistic Regression with NAGD Optimization
class LogisticRegression_NAGD:
    def __init__(self, learning_rate: float = 0.05, iteration: int = 1000, random_state: int = 1234):
        self.learning_rate = learning_rate
        self.iteration     = iteration
        self.random_state  = random_state

    def fit(self, X, y):
        X_ = np.concatenate([X, np.ones([len(X), 1])], axis=1)  #this is to incorporate bias term

        np.random.seed(self.random_state)
        self.w    = np.random.randn(np.shape(X_)[1],1)

        w_prev = np.copy(self.w)
    
        self.loss = []
        ### NAGD algorithm
        for t in range(self.iteration):

            ### NAGD with MSE Loss
            grad_temp = (-1 / len(X_)) * np.dot(X_.T, (y - logistic(X_, self.w)))
            gamma = 0.9
            grad_prev = 0
            grad_prev = gamma * grad_prev + self.learning_rate * grad_temp
            w_temp = self.w - grad_prev + ((t-1)/(t+2)) * (self.w - w_prev)
            w_prev = np.copy(self.w)
            self.w = np.copy(w_temp)

            self.loss.append(loss_MSE(X_, y, self.w))  #this is to track loss behavior (do not remove it)
    
    def predict(self, X):
        return logistic(np.concatenate([X, np.ones([len(X), 1])], axis=1), self.w)
