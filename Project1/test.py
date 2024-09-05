import numpy as np
import pandas as pd

from models import NaiveBayesClassifier

# import data
tr_df = pd.read_csv('./train_data.csv')
te_df = pd.read_csv('./train_data.csv')

# split feature/labels
feat_list = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capitalgain', 'capitalloss', 'hoursperweek']
label     = 'class'

tr_X      = np.asarray(tr_df[feat_list])
tr_y      = np.asarray(tr_df[label])

te_X      = np.asarray(te_df[feat_list])
te_y      = np.asarray(te_df[label])


model = NaiveBayesClassifier()

model.fit(tr_X,tr_y)

print(model.conditional_prob)
print(np.mean(model.predict(te_X) == te_y))
