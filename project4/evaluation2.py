import numpy as np
import pandas as pd

from models_boosting import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### ===============================================================
### EXPERIMENT 2
### Compare Boosting with max_depth=1 vs max_depth=None  
### ===============================================================


### TRAINING
### import training data
tr_data = pd.read_csv('./dataset/train_data.csv')
tr_data = tr_data[tr_data['label'].isin([3,8])].reset_index(drop=True) # change the task into binary classifiercation (i.e. 3 vs 8)

tr_y = np.asarray(tr_data[['label']]).reshape([-1])
tr_X = np.asarray(tr_data.drop(columns=['label']))
tr_y[tr_y == 3] = -1
tr_y[tr_y == 8] = +1

# base and boosting classifiers with max_depth=1
base_model1 = DecisionTreeClassifier(max_depth=1)
model1 = AdaBoostClassifier(n_estimators=1000, max_depth=1)

base_model1.fit(tr_X,tr_y)
model1.fit(tr_X, tr_y)


# base and boosting classifiers with max_depth=None
base_model2 = DecisionTreeClassifier(max_depth=None)
model2 = AdaBoostClassifier(n_estimators=1000, max_depth=None)

base_model2.fit(tr_X,tr_y)
model2.fit(tr_X, tr_y)




# ### ===============================================================
# ### Test data and code will be used for grading. Intentially commented out.
# ### ===============================================================

# ### TESTING
# ### import testing data
te_data = pd.read_csv('./dataset/test_data.csv')
te_data = te_data[te_data['label'].isin([3,8])].reset_index(drop=True) # change the task into binary classifiercation (i.e. 3 vs 8)

te_y = np.asarray(te_data[['label']]).reshape([-1])
te_X = np.asarray(te_data.drop(columns=['label']))
te_y[te_y == 3] = -1
te_y[te_y == 8] = +1

# base and bagging classifiers with max_depth=None
base_pred1 = base_model1.predict(te_X)
pred1      = model1.predict(te_X)

# base and bagging classifiers with max_depth=5
base_pred2 = base_model2.predict(te_X)
pred2      = model2.predict(te_X)

print("ACC Weak Learner (max_depth=1): {:.4f}".format(accuracy_score(te_y, base_pred1)) )
print("ACC Boosting Classifier (max_depth=1): {:.4f}".format(accuracy_score(te_y, pred1)) )

print("ACC Weak Learner (max_depth=5): {:.4f}".format(accuracy_score(te_y, base_pred2)) )
print("ACC Boosting Classifier (max_depth=5): {:.4f}".format(accuracy_score(te_y, pred2)) )