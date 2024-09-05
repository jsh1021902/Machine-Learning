import numpy as np
import pandas as pd

from models_bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### ===============================================================
### EXPERIMENT 1
### Compare Bagging with max_depth=None vs max_depth=5  
### ===============================================================


### TRAINING
### import training data
tr_data = pd.read_csv('./dataset/train_data.csv')

tr_y = np.asarray(tr_data[['label']]).reshape([-1])
tr_X = np.asarray(tr_data.drop(columns=['label']))


# base and bagging classifiers with max_depth=None
base_model1 = DecisionTreeClassifier(max_depth=None)
model1 = BaggingClassifier(n_estimators=100, max_depth=None)

base_model1.fit(tr_X,tr_y)
model1.fit(tr_X, tr_y)

# base and bagging classifiers with max_depth=5
base_model2 = DecisionTreeClassifier(max_depth=5)
model2 = BaggingClassifier(n_estimators=100, max_depth=5)

base_model2.fit(tr_X,tr_y)
model2.fit(tr_X, tr_y)



# ### ===============================================================
# ### Test data and code will be used for grading. Intentially commented out.
# ### ===============================================================

# ### TESTING
## import testing data
te_data = pd.read_csv('./dataset/test_data.csv')

te_y = np.asarray(te_data[['label']]).reshape([-1])
te_X = np.asarray(te_data.drop(columns=['label']))

# base and bagging classifiers with max_depth=None
base_pred1 = base_model1.predict(te_X)
pred1      = model1.predict(te_X)

# base and bagging classifiers with max_depth=5
base_pred2 = base_model2.predict(te_X)
pred2      = model2.predict(te_X)


print("ACC Base Classifier (max_depth=None): {:.4f}".format(accuracy_score(te_y, base_pred1)) )
print("ACC Bagging Classifier (max_depth=None): {:.4f}".format(accuracy_score(te_y, pred1)) )

print("ACC Base Classifier (max_depth=5): {:.4f}".format(accuracy_score(te_y, base_pred2)) )
print("ACC Bagging Classifier (max_depth=5): {:.4f}".format(accuracy_score(te_y, pred2)) )