import csv
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt

df = pd.read_csv("data/flipped_data/scaled_merged_features_and_flipped_labels.csv", engine="python")
df = df.drop(['party', 'ID', 'STATE', 'SC', 'CD'], 1)
df[['CVLLBFRC', 'MANUF', 'MDNINCM', 'PORT', 'VETERANS', 'TRANSPRT']] = df[['CVLLBFRC', 'MANUF', 'MDNINCM', 'PORT', 'VETERANS', 'TRANSPRT']].fillna(0)

clf_knn = KNeighborsClassifier()
clf_id3_underfit = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
clf_id3 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=8)
clf_id3_overfit = tree.DecisionTreeClassifier(criterion="entropy")
clf_cart = tree.DecisionTreeClassifier(max_depth=18)
clf_bayes = GaussianNB()
clf_rbf = SVC()
clf_forest = RandomForestClassifier()
clf_boost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), random_state=21)
clf_mlp = MLPClassifier(hidden_layer_sizes=(50,50,), max_iter=1000, tol=0.001, random_state=42)

# test/train split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print(Y_train)
print("No Sampling: Label = flip:", sum(Y_train == 1))
print("No Sampling: Label = not flip:", sum(Y_train == 0))

# initialize SMOTE oversampling algorithm; oversamples all rows with label not in majority
sm = SMOTE(random_state=2)
X_train_OS, Y_train_OS = sm.fit_sample(X_train, Y_train)
print("SMOTE Oversampling: Label = flip:", sum(Y_train_OS == 1))
print("SMOTE Oversampling: Label = not flip:", sum(Y_train_OS == 0))

# initialize near miss undersampling algorithm
nm = NearMiss()
X_train_US, Y_train_US = nm.fit_sample(X_train, Y_train)
print("SMOTE Undersampling: Label = flip:", sum(Y_train_US == 1))
print("SMOTE Undersampling: Label = not flip:", sum(Y_train_US == 0))

for feature in df.columns:
  pass