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
import statistics
import operator

# optimizing adaboost

df = pd.read_csv("./data/flipped_data/scaled_merged_features_and_flipped_labels.csv", engine="python")
df = df.drop(['party', 'ID', 'STATE', 'SC', 'CD'], 1)
df[['CVLLBFRC', 'MANUF', 'MDNINCM', 'PORT', 'VETERANS', 'TRANSPRT']] = df[['CVLLBFRC', 'MANUF', 'MDNINCM', 'PORT', 'VETERANS', 'TRANSPRT']].fillna(0)
df = df[df['prev_party'].notna()]

y = df['flip']
x = df.drop(['flip'], 1)

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

# clf_boost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), random_state=21)


def testing_random_forest():
  max_depths = []
  depths = []

  accuracies_test = []
  auc_test = []
  for max_d in range(1, 101):
      clf_forest = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=8), n_estimators=max_d)
      clf_forest.fit(X_train_OS, Y_train_OS)
      print(f"{max_d} n_estimators")
      accuracy = metrics.accuracy_score(Y_test, clf_forest.predict(X_test))
      print(f"\tAccuracy: {accuracy}")
      accuracies_test.append(accuracy)
      auc_test.append(metrics.roc_auc_score(Y_test, clf_forest.predict(X_test)))
      max_depths.append(max_d)

  plt.plot(max_depths, accuracies_test)
  plt.title("n_estimators vs Test Accuracy")
  plt.xlabel('n_estimators')
  plt.ylabel('Test Accuracy')
  plt.show()
  plt.title("n_estimators vs Test AUC")
  plt.plot(max_depths, auc_test)
  plt.xlabel('n_estimators')
  plt.ylabel('Test AUC')
  plt.show()

testing_random_forest()


