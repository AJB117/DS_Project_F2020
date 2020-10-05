#!/usr/bin/python3
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
accuracy = [ [] for _ in range(10) ]
for _ in range(1):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # print(X_train.columns[3])

    # print(X_train)

    # print(np.where(x.applymap(lambda x: x == '')))

    clf_knn = KNeighborsClassifier()
    clf_id3_underfit = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
    clf_id3 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=18)
    clf_id3_overfit = tree.DecisionTreeClassifier(criterion="entropy")
    clf_cart = tree.DecisionTreeClassifier(max_depth=18)
    clf_bayes = GaussianNB()
    clf_rbf = SVC()
    clf_forest = RandomForestClassifier()
    clf_boost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2), random_state=21)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(50,50,), max_iter=1000, tol=0.001, random_state=42)

    # model train and test
    clf_knn.fit(X_train, y_train)
    clf_id3_underfit.fit(X_train, y_train)
    clf_id3.fit(X_train, y_train)
    clf_id3_overfit.fit(X_train, y_train)
    clf_cart.fit(X_train, y_train)
    clf_bayes.fit(X_train, y_train)
    clf_rbf.fit(X_train, y_train)
    clf_forest.fit(X_train, y_train)
    clf_boost.fit(X_train, y_train)
    clf_mlp.fit(X_train, y_train)

    print("Accuracy:")
    print("- KNN:", metrics.accuracy_score(y_test, clf_knn.predict(X_test)))
    print("- ID3 (underfitting):", metrics.accuracy_score(y_test, clf_id3_underfit.predict(X_test)))
    print("- ID3:", metrics.accuracy_score(y_test, clf_id3.predict(X_test)))
    print("- ID3 (overfitting):", metrics.accuracy_score(y_test, clf_id3_overfit.predict(X_test)))
    print("- CART:", metrics.accuracy_score(y_test, clf_cart.predict(X_test)))
    print("- Naive Bayes:", metrics.accuracy_score(y_test, clf_bayes.predict(X_test)))
    print("- RBF Kernel SVC:", metrics.accuracy_score(y_test, clf_rbf.predict(X_test)))
    print("- Random Forest:", metrics.accuracy_score(y_test, clf_forest.predict(X_test)))
    print("- AdaBoost:", metrics.accuracy_score(y_test, clf_boost.predict(X_test)))
    print("- MLP:", metrics.accuracy_score(y_test, clf_mlp.predict(X_test)))

    print("Confusion Matrices:")
    print("- KNN:\n", metrics.confusion_matrix(y_test, clf_knn.predict(X_test)))
    print("- ID3 (underfitting):\n", metrics.confusion_matrix(y_test, clf_id3_underfit.predict(X_test)))
    print("- ID3:\n", metrics.confusion_matrix(y_test, clf_id3.predict(X_test)))
    print("- ID3 (overfitting):\n", metrics.confusion_matrix(y_test, clf_id3_overfit.predict(X_test)))
    print("- CART:\n", metrics.confusion_matrix(y_test, clf_cart.predict(X_test)))
    print("- Naive Bayes:\n", metrics.confusion_matrix(y_test, clf_bayes.predict(X_test)))
    print("- RBF Kernel SVC:\n", metrics.confusion_matrix(y_test, clf_rbf.predict(X_test)))
    print("- Random Forest:\n", metrics.confusion_matrix(y_test, clf_forest.predict(X_test)))
    print("- AdaBoost:\n", metrics.confusion_matrix(y_test, clf_boost.predict(X_test)))
    print("- MLP:\n", metrics.confusion_matrix(y_test, clf_mlp.predict(X_test)))
    
    accuracy[0].append(metrics.accuracy_score(y_test, clf_knn.predict(X_test)))
    accuracy[1].append(metrics.accuracy_score(y_test, clf_id3_underfit.predict(X_test)))
    accuracy[2].append(metrics.accuracy_score(y_test, clf_id3.predict(X_test)))
    accuracy[3].append(metrics.accuracy_score(y_test, clf_id3_overfit.predict(X_test)))
    accuracy[4].append(metrics.accuracy_score(y_test, clf_cart.predict(X_test)))
    accuracy[5].append(metrics.accuracy_score(y_test, clf_bayes.predict(X_test)))
    accuracy[6].append(metrics.accuracy_score(y_test, clf_rbf.predict(X_test)))
    accuracy[7].append(metrics.accuracy_score(y_test, clf_forest.predict(X_test)))
    accuracy[8].append(metrics.accuracy_score(y_test, clf_boost.predict(X_test)))
    accuracy[9].append(metrics.accuracy_score(y_test, clf_mlp.predict(X_test)))

    '''clf_id3 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=d)
    clf_cart = tree.DecisionTreeClassifier(max_depth=d)

    clf_id3.fit(X_train, y_train)
    clf_cart.fit(X_train, y_train)

    
    accuracy[0].append(d)

    print(f'- ID3: max_depth={d}', metrics.accuracy_score(y_test, clf_id3.predict(X_test)))
    accuracy[1].append(metrics.accuracy_score(y_test, clf_id3.predict(X_test)))

    print(f'- CART: max_depth={d}', metrics.accuracy_score(y_test, clf_cart.predict(X_test)))
    accuracy[2].append(metrics.accuracy_score(y_test, clf_cart.predict(X_test)))'''

    



# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

with open(f'output/output-{dt_string}.csv',"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(accuracy)