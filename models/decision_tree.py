#!/usr/local/bin/python3
import math
import pandas as pd
from sklearn import tree

df = pd.read_csv('../data/integrated_features_and_labels.csv', sep=',')
bad_features = set()

for sample in df.values:
    for i, featurevalue in enumerate(sample):
        if featurevalue == ' ' or (
            type(featurevalue) != str and math.isnan(featurevalue)
        ):
            bad_features.add(i)

present_values = []

for sample in df.values:
    present_values.append([
        _ for i, _ in enumerate(sample) if i not in bad_features and i != 0
    ])

df = present_values

le = LabelEncoder()
df['PARTY'] = le.fit_transform(df['PARTY'].values)
y = df['PARTY']
x = df.drop(['PARTY'], 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
