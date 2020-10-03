import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

df = pd.read_csv("./data/integrated_features_and_labels.csv", engine="python")

# some (heavy) preprocessing
le = LabelEncoder()
print(df['PARTY'])
df['PARTY'] = le.fit_transform(df['PARTY'].values)
df = df.drop(['REDIST', 'ACEREGIO', 'CONSTRUC', 'CVLLBRFR', 'CANDIDATEVOTES', 'TOTALVOTES', 'ID', 'PORT', 'YEAR', 'STATE', 
              'PREVID', 'PREVCANDVO', 'PREVTOTVO', 'FIPSTATE', 'SC', 'CD', 'PREVPARTY', 'NUCPLANT', 'POPSQMI', 'MDNINCM',
              'MANUF', 'WHLRETL', 'VETERANS', 'VABEDS', 'TRANSPRT'], 1)

# test/train split
y = df['PARTY']
# print(y)
x = df.drop(['PARTY'], 1)
# print(x.corr(method="pearson").to_string())
# print(np.where(pd.isnull(df)))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(X_train.columns[3])

# print(X_train)

# print(np.where(x.applymap(lambda x: x == '')))

# model train and test
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)

print(f"Tree Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"KNN Accuracy: {metrics.accuracy_score(y_test, clf_knn.predict(X_test))}")

# cols = list(x.columns)
# print(cols)