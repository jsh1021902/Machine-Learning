import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import NeuralNetwork

import warnings

#suppress warnings
warnings.filterwarnings('ignore')


### ===============================================================
### EXPERIMENT
### Train and Test Neural Networks
### ===============================================================


### TRAINING
# import training data
tr_data = pd.read_csv('./dataset/train_data.csv')
tr_y = np.asarray(tr_data[['label']])
tr_X = np.asarray(tr_data.drop(columns=['label']))

# Hyper-parameters: Netwrok Architecture
input_size  = tr_X.shape[1]
hidden_size = 100
output_size = 1

# Hyper-parameters: Optimization
iterations    = 5000
learning_rate = 1e-3
mb_size       = 32


model = NeuralNetwork(input_size, hidden_size, output_size)
losses        = model.train(tr_X, tr_y, iterations, learning_rate, mb_size)



### PLOTTING
plt.figure(figsize=[8,5])
plt.plot(np.arange(0, len(losses))*500, losses, color='C0')
plt.xlabel('iteration', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('BCE Loss', fontsize=14)
plt.grid()
plt.savefig('./fig_itr_vs_loss.png')
plt.show()
plt.close()

# ## ===============================================================
# ## Test data and code will be used for grading. Intentially commented out.
# ## ===============================================================

# ### TESTING
# # import testing data
# te_data = pd.read_csv('./dataset/test_data.csv')
# te_y = np.asarray(te_data[['label']])
# te_X = np.asarray(te_data.drop(columns=['label']))

# pred = model.forward(te_X)

# print("Accuracy : {:.4f}".format(np.mean(te_y == (pred > 0.5).astype(float))))
