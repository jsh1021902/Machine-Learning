import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from models_solution import LogisticRegression_Kernelized
from models import LogisticRegression_Kernelized

### ===============================================================
### EXPERIMENT
###     - Evaluate accruacy of the kernelized logistic regression with ridge penalty
###     - Visualize the decision boundary
### ===============================================================


### TRAINING
# import training data
tr_data = pd.read_csv('./dataset/train_data.csv')
tr_Y = np.asarray(tr_data[['label']])
tr_X = np.asarray(tr_data.drop(columns=['label']))

### Do not change lr_rate, gamma, and lmbda (since training of the kernelized LR is unstable)
lr_rate      = 0.001
max_itr      = 1000
gamma        = 0.1
lmbda        = 1e-2

model        = LogisticRegression_Kernelized(learning_rate=lr_rate, gamma=gamma, lmbda=lmbda, iteration=max_itr)
model.fit(tr_X, tr_Y)


# ### ===============================================================
# ### Test data and code will be used for grading. Intentially commented out.
# ### ===============================================================
# from sklearn.metrics import accuracy_score

# ### TESTING
# # import testing data
# te_data = pd.read_csv('./dataset/test_data.csv')
# te_Y = np.asarray(te_data[['label']])
# te_X = np.asarray(te_data.drop(columns=['label']))

# pred = model.predict(te_X)


# print("Accuracy LogisticRegression_CE  : {:.4f}".format(accuracy_score((te_Y > 0.5).astype(float), (pred > 0.5).astype(float)) ))



### ===============================================================
### Visualize decision boundary of the learned model
### ===============================================================
tmp_x0 = np.linspace(-1.5, 2.5, 500)
tmp_x1 = np.linspace(-1.0, 1.5, 500)

xx, yy = np.meshgrid(tmp_x0, tmp_x1)

mesh   = np.concatenate(
    [xx.reshape([-1, 1]), 
     yy.reshape([-1, 1])], axis=1
)

pred_mesh = (model.predict(mesh) > 0.5).astype(float)
zz        = pred_mesh.reshape(xx.shape)

plt.figure(figsize=[10,7])

idx1 = tr_Y[:, 0] == 1
idx0 = tr_Y[:, 0] == -1

plt.contourf(xx, yy, zz, cmap='RdBu', alpha=0.3)

plt.scatter(tr_X[idx1, 0], tr_X[idx1, 1], alpha=0.5, edgecolor='gray', color='C0', s=80)
plt.scatter(tr_X[idx0, 0], tr_X[idx0, 1], alpha=0.5, edgecolor='gray', color='C3', s=80)

plt.xlabel(r'$X_{0}$', fontsize=15)
plt.ylabel(r'$X_{1}$', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid()
plt.savefig('./decisionboundary_KernelizedLR.png')
plt.show()
plt.close()