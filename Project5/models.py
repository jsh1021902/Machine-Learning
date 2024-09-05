import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size)
        self.W3 = np.random.randn(self.hidden_size, self.output_size)

        # Initialize biases
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.hidden_size))
        self.b3 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0]  = 1
        return x

    def cross_entropy(self, y, y_pred):
        return np.mean(-(y * np.log(y_pred+1e-8) + (1.-y) * np.log(1.-y_pred+1e-8)))

    def sample_batch(self, X, y, mb_size=32):
        idx =  np.random.choice(len(X), size=mb_size, replace=False) # you may use "np.random.choice"
        return X[idx], y[idx]

    def forward(self, X):
        # Forward propagation

        # hidden layer 1
        self.z1  = np.dot(X, self.W1) + self.b1
        self.h1  = self.relu(self.z1) # hint: this is the output after non-linear activation (i.e., h1 = g(z1))

        # hidden layer 2
        self.z2  = np.dot(self.h1, self.W2) + self.b2
        self.h2  = self.relu(self.z2)

        # output layer 
        self.z3  = np.dot(self.h2, self.W3) + self.b3
        self.out = self.sigmoid(self.z3)

        return self.out

    def backward(self, X, y, y_pred, learning_rate):
        m   = float(y.shape[0])
        
        # Calculate gradients
        # output layer (this is a hint)
        dz3 = y_pred - y
        dW3 = (1/m) * np.dot(self.h2.T, dz3)
        db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
        
        # hidden layer 2
        dh2 = np.dot(dz3, self.W3.T)
        dz2 = dh2 * self.relu_derivative(self.z2)
        dW2 = (1 / m) * np.dot(self.h1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        
        # hidden layer 1
        dh1 = np.dot(dz2, self.W2.T)
        dz1 = dh1 * self.relu_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, iterations=1e4, learning_rate=1e-3, mb_size=32):
        losses = []

        for itr in range(iterations):
            X_mb, y_mb = self.sample_batch(X, y, mb_size)
            y_pred     = self.forward(X_mb)
            self.backward(X_mb, y_mb, y_pred, learning_rate)

            if (itr+1) % 500 == 0:
                # Forward propagation
                y_pred = self.forward(X)
                # Calculate loss
                loss   = self.cross_entropy(y, y_pred)                
                print(f"iteration: {itr+1}, Loss: {loss:.4f}")
                losses.append(loss)

        return losses