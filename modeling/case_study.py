import csv
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import statistics

df = pd.read_csv("../data/flipped_data/scaled_merged_features_and_flipped_labels.csv", engine="python")
df = df.drop(['party', 'ID', 'STATE', 'SC', 'CD'], 1)

# df_case_study = pd.read_csv("data/flipped_data/1978_case_study.csv", engine="python")
df_case_study = pd.read_csv("data/flipped_data/1978_case_study.csv", engine="python")
df_case_study = df_case_study.drop(['party', 'ID', 'STATE', 'SC', 'CD'], axis=1)

# test/train split
y = df['flip']
x = df.drop(['flip'], 1)

y_cs = df_case_study['flip'].values.tolist()
x_cs = df_case_study.drop(['flip'], 1)
print(len(y_cs))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print(Y_train)
print("No Sampling: Label = flip:", sum(Y_train == 1))
print("No Sampling: Label = not flip:", sum(Y_train == 0))

# initialize SMOTE oversampling algorithm; oversamples all rows with label not in majority
sm = SMOTE(random_state=2)
X_train_OS, Y_train_OS = sm.fit_sample(X_train, Y_train)
print("SMOTE Oversampling: Label = flip:", sum(Y_train_OS == 1))
print("SMOTE Oversampling: Label = not flip:", sum(Y_train_OS == 0))

clf_boost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), random_state=21)
clf_boost.fit(X_train_OS, Y_train_OS)

for index, data_object in enumerate(x_cs.values.tolist()):
  db = data_object
  data_object = np.array(data_object)
  if (y_cs[index] != clf_boost.predict(data_object.reshape(1, -1))):
    print(index)

