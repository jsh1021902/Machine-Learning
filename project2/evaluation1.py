import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import LogisticRegression_CE, LogisticRegression_MSE

### ===============================================================
### EXPERIMENT 1
### Compare CE Loss vs MSE Loss
### ===============================================================


### TRAINING
# import training data
tr_data = pd.read_csv('./dataset/train_data.csv')
tr_y = np.asarray(tr_data[['label']])
tr_X = np.asarray(tr_data.drop(columns=['label']))


lr_rate      = 0.05
max_itr      = 1000
seed         = 1234

model1 = LogisticRegression_CE(learning_rate=lr_rate, iteration=max_itr, random_state=seed)
model2 = LogisticRegression_MSE(learning_rate=lr_rate, iteration=max_itr, random_state=seed)

model1.fit(tr_X,tr_y)
model2.fit(tr_X,tr_y)


### PLOTTING
plt.figure(figsize=[10,5])
plt.plot(np.arange(len(model1.loss)), np.asarray(model1.loss), color='C0')
plt.xlabel('iteration', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('GD - CE Loss', fontsize=14)
plt.grid()
plt.savefig('./fig_itr_vs_CELoss.png')
plt.show()
plt.close()

plt.figure(figsize=[10,5])
plt.plot(np.arange(len(model2.loss)), np.asarray(model2.loss), color='C3')
plt.xlabel('iteration', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('GD - MSE Loss', fontsize=14)
plt.grid()
plt.savefig('./fig_itr_vs_MSELoss.png')
plt.show()
plt.close()


### ===============================================================
### Test data and code will be used for grading. Intentially commented out.
### ===============================================================

# ### TESTING
# # import testing data
# te_data = pd.read_csv('./dataset/test_data.csv')
# te_y = np.asarray(te_data[['label']])
# te_X = np.asarray(te_data.drop(columns=['label']))

# pred1 = model1.predict(te_X)
# pred2 = model2.predict(te_X)

# print("Accuracy LogisticRegression_CE  : {:.4f}".format(np.mean(te_y == (pred1 > 0.5).astype(float))))
# print("Accuracy LogisticRegression_MSE : {:.4f}".format(np.mean(te_y == (pred2 > 0.5).astype(float))))
