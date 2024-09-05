import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import LogisticRegression_CE, LogisticRegression_NAGD

### ===============================================================
### EXPERIMENT 2
###     - Compare GD vs NAGD
###     - NAGD will converge much faster with smaller number of iterations
### ===============================================================


### TRAINING
# import training data
tr_data = pd.read_csv('./dataset/train_data.csv')
tr_y = np.asarray(tr_data[['label']])
tr_X = np.asarray(tr_data.drop(columns=['label']))


lr_rate      = 0.05
max_itr      = 500 #now you will set a very small number of iterations
seed         = 1234

model1 = LogisticRegression_CE(learning_rate=lr_rate, iteration=max_itr, random_state=seed)
model3 = LogisticRegression_NAGD(learning_rate=lr_rate, iteration=max_itr, random_state=seed)

model1.fit(tr_X,tr_y)
model3.fit(tr_X,tr_y)


### PLOTTING
plt.figure(figsize=[10,5])
plt.plot(np.arange(len(model1.loss)), np.asarray(model1.loss), color='C0')
plt.plot(np.arange(len(model3.loss)), np.asarray(model3.loss), color='C1')

plt.xlabel('iteration', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('GD vs NAGD - CE Loss', fontsize=14)
plt.legend(['GD', 'NAGD'], fontsize=13)
plt.grid()
plt.savefig('./fig_itr_vs_GD_NAGD.png')
plt.show()
plt.close()



### ===============================================================
### Test data and code will be used for grading. Intentially commented out.
### ===============================================================

### TESTING 
# import testing data
#te_data = pd.read_csv('./dataset/test_data.csv') 
#te_y = np.asarray(te_data[['label']])
#te_X = np.asarray(te_data.drop(columns=['label']))

#pred1 = model1.predict(te_X)
#pred3 = model3.predict(te_X)

#print("Accuracy LogisticRegression_CE  : {:.4f}".format(np.mean(te_y == (pred1 > 0.5).astype(float))))
#print("Accuracy LogisticRegression_NAGD: {:.4f}".format(np.mean(te_y == (pred3 > 0.5).astype(float))))
