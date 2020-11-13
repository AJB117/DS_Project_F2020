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
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import statistics

df = pd.read_csv("../data/flipped_data/scaled_merged_features_and_flipped_labels.csv", engine="python")
df = df.drop(['party', 'ID', 'STATE', 'SC', 'CD'], 1)

# feature/label split
y = df['flip']
x = df.drop(['flip'], 1)

num_trials = 1
results = {}
all_metrics = ["accuracy", "TP", "FN", "FP", "TN", "precision", "recall", "f1", "auc"]
features = df.columns
for feature in features:
    results.update({feature: {} })
    for metric in all_metrics:
        results[feature].update({metric: [] })
now = datetime.now()

clf_boost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), n_estimators=45, random_state=21)
original_df = df

# loops once for each trial
for trial in range(num_trials):
    # Print
    for feature in features:
        df = original_df.drop([feature], 'columns')

        print(f'TRIAL {trial}')

        # test/train split
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
        print(Y_train)
        print("No Sampling: Label = flip:", sum(Y_train == 1))
        print("No Sampling: Label = not flip:", sum(Y_train == 0))
        
        print(' OVERSAMPLING')
        # initialize SMOTE oversampling algorithm; oversamples all rows with label not in majority
        sm = SMOTE(random_state=2)
        X_train_OS, Y_train_OS = sm.fit_sample(X_train, Y_train)
        print("SMOTE Oversampling: Label = flip:", sum(Y_train_OS == 1))
        print("SMOTE Oversampling: Label = not flip:", sum(Y_train_OS == 0))
        new_now = datetime.now()
        elapsed = new_now - now
        print(f'trial {trial} feature {feature} ran in {elapsed} time')
        now = new_now

        # model train and test
        clf_boost.fit(X_train_OS, Y_train_OS)

        # Test
        metric = "accuracy"
        results[feature][metric].append(metrics.accuracy_score(Y_test, clf_boost.predict(X_test)))

        confusion_matrix = [["TP", "FN"],
                            ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                metric = confusion_matrix[i][j]
                results[feature][metric].append(metrics.confusion_matrix(Y_test, clf_boost.predict(X_test))[i][j])
        
        print(results)

        metric = "precision"
        val = results[feature]["TP"][-1] / (results[feature]["TP"][-1]+results[feature]["FP"][-1])
        results[feature][metric].append(val)

        metric = "recall"
        results[feature][metric].append(results[feature]["TP"][-1] / (results[feature]["TP"][-1]+results[feature]["FN"][-1]))

        metric = "f1"
        results[feature][metric].append(2*results[feature]["precision"][-1]*results[feature]["recall"][-1] / (results[feature]["precision"][-1]+results[feature]["recall"][-1]))
        
        metric = "auc"
        results[feature][metric].append(metrics.roc_auc_score(Y_test, clf_boost.predict(X_test)))

    

# flatten results to a 2D array
metric_labels = []
for metric in all_metrics:
    metric_labels += [f'Mean {metric}', f'S.D. {metric}']
data = [ ["Feature"] + metric_labels ]
for feature in features:
    data_row = [feature]
    for metric in all_metrics:
        data_row += [ statistics.mean(results[feature][metric]), statistics.stdev(results[feature][metric])]
    data.append(data_row)

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

with open(f'../output/feature_interpretation/interpretation-output-{dt_string}.csv',"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(data)