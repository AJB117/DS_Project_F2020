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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import statistics

df = pd.read_csv("./data/flipped_data/scaled_merged_features_and_flipped_labels.csv", engine="python")

### CHANGE THIS WITH NEW DATA
# some (heavy) preprocessing
# le = LabelEncoder()
# df['PARTY'] = le.fit_transform(df['PARTY'].values)
# df = df.drop(['REDIST', 'ACEREGIO', 'CONSTRUC', 'CVLLBRFR', 'CANDIDATEVOTES', 'TOTALVOTES', 'ID', 'PORT', 'YEAR', 'STATE', 
#               'PREVID', 'PREVCANDVO', 'PREVTOTVO', 'FIPSTATE', 'SC', 'CD', 'PREVPARTY', 'NUCPLANT', 'POPSQMI', 'MDNINCM',
#               'MANUF', 'WHLRETL', 'VETERANS', 'VABEDS', 'TRANSPRT'], 1)
# print(df['PARTY'])

df = df.drop(['party', 'ID', 'STATE', 'SC', 'CD'], 1)
df[['CVLLBFRC', 'MANUF', 'MDNINCM', 'PORT', 'VETERANS', 'TRANSPRT' ]] = df[['CVLLBFRC', 'MANUF', 'MDNINCM', 'PORT', 'VETERANS', 'TRANSPRT']].fillna(0)

# df = df[df['prev_party'].isna() & df['prev_candidatevotes'].isna() & df['prev_totalvotes'].isna() & df['prev_winratio'].isna() & df['flip'].isna()]
df = df[df['prev_party'].notna()]

print(df.isna().any())
# feature/label split
y = df['flip']
x = df.drop(['flip'], 1)

num_trials = 10
results = {}
samplings = [ "STANDARD", "UNDER", "OVER"]
models = [ "KNN", "ID3 (Underfitting)", "ID3", "ID3 (Overfitting)",
        "CART", "Naive Bayes", "RBF Kernel SVC", "Random Forest", "AdaBoost", "MLP" ]
all_metrics = ["accuracy", "TP", "FN", "FP", "TN", "precision", "recall", "f1"]
# test_dict = { models[0]: {all_metrics[0]: [] } }
# print(test_dict)
for sampling in samplings:
    results.update({sampling: {} })
    for model in models:
        results[sampling].update({model: {} })
        for metric in all_metrics:
            results[sampling][model].update({metric: [] })
# print(results)
now = datetime.now()


# Initialize
clf_knn = KNeighborsClassifier()
clf_id3_underfit = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
clf_id3 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=8)
clf_id3_overfit = tree.DecisionTreeClassifier(criterion="entropy")
clf_cart = tree.DecisionTreeClassifier(max_depth=18)
clf_bayes = GaussianNB()
clf_rbf = SVC()
clf_forest = RandomForestClassifier(n_estimators=15, max_depth=12)
clf_boost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), random_state=21)
clf_mlp = MLPClassifier(hidden_layer_sizes=(50,50,), max_iter=1000, tol=0.001, random_state=42)

# loops once for each trial
for trial in range(num_trials):
    # Print
    for sampling in samplings:
        
        for sample_key, sampling_dict in results.items():
            for model_key, model_dict in sampling_dict.items():
                for metric_key, trial_results in model_dict.items():
                    print(f'{sample_key} {model_key} {metric_key}: {trial_results}')

        # print(X_train.columns[3])

        # print(X_train)

        # print(np.where(x.applymap(lambda x: x == '')))
        print(f'TRIAL {trial}')

        # test/train split
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
        print(Y_train)
        print("No Sampling: Label = flip:", sum(Y_train == 1))
        print("No Sampling: Label = not flip:", sum(Y_train == 0))
        
        # Sample and train
        if sampling == "STANDARD":
            print(" STANDARD SAMPLING")
             # model train and test
            clf_knn.fit(X_train, Y_train)
            clf_id3_underfit.fit(X_train, Y_train)
            clf_id3.fit(X_train, Y_train)
            clf_id3_overfit.fit(X_train, Y_train)
            clf_cart.fit(X_train, Y_train)
            clf_bayes.fit(X_train, Y_train)
            clf_rbf.fit(X_train, Y_train)
            clf_forest.fit(X_train, Y_train)
            clf_boost.fit(X_train, Y_train)
            clf_mlp.fit(X_train, Y_train)            
        elif sampling == "UNDER":
            print(' UNDERSAMPLING')
            # initialize near miss undersampling algorithm
            nm = NearMiss()
            X_train_US, Y_train_US = nm.fit_sample(X_train, Y_train)
            print("SMOTE Undersampling: Label = flip:", sum(Y_train_US == 1))
            print("SMOTE Undersampling: Label = not flip:", sum(Y_train_US == 0))

            # model train and test
            clf_knn.fit(X_train_US, Y_train_US)
            clf_id3_underfit.fit(X_train_US, Y_train_US)
            clf_id3.fit(X_train_US, Y_train_US)
            clf_id3_overfit.fit(X_train_US, Y_train_US)
            clf_cart.fit(X_train_US, Y_train_US)
            clf_bayes.fit(X_train_US, Y_train_US)
            clf_rbf.fit(X_train_US, Y_train_US)
            clf_forest.fit(X_train_US, Y_train_US)
            clf_boost.fit(X_train_US, Y_train_US)
            clf_mlp.fit(X_train_US, Y_train_US)
        elif sampling == "OVER":
            print(' OVERSAMPLING')
            # initialize SMOTE oversampling algorithm; oversamples all rows with label not in majority
            sm = SMOTE(random_state=2)
            X_train_OS, Y_train_OS = sm.fit_sample(X_train, Y_train)
            print("SMOTE Oversampling: Label = flip:", sum(Y_train_OS == 1))
            print("SMOTE Oversampling: Label = not flip:", sum(Y_train_OS == 0))

            # model train and test
            clf_knn.fit(X_train_OS, Y_train_OS)
            clf_id3_underfit.fit(X_train_OS, Y_train_OS)
            clf_id3.fit(X_train_OS, Y_train_OS)
            clf_id3_overfit.fit(X_train_OS, Y_train_OS)
            clf_cart.fit(X_train_OS, Y_train_OS)
            clf_bayes.fit(X_train_OS, Y_train_OS)
            clf_rbf.fit(X_train_OS, Y_train_OS)
            clf_forest.fit(X_train_OS, Y_train_OS)
            clf_boost.fit(X_train_OS, Y_train_OS)
            clf_mlp.fit(X_train_OS, Y_train_OS)
        else:
            print(f'ERROR, FOUND NO MATCH FOR {sampling}')

            

        # Test
        metric = "accuracy"
        results[sampling]["KNN"][metric].append(metrics.accuracy_score(Y_test, clf_knn.predict(X_test)))
        results[sampling]["ID3 (Underfitting)"][metric].append(metrics.accuracy_score(Y_test, clf_id3_underfit.predict(X_test)))
        results[sampling]["ID3"][metric].append(metrics.accuracy_score(Y_test, clf_id3.predict(X_test)))
        results[sampling]["ID3 (Overfitting)"][metric].append(metrics.accuracy_score(Y_test, clf_id3_overfit.predict(X_test)))
        results[sampling]["CART"][metric].append(metrics.accuracy_score(Y_test, clf_cart.predict(X_test)))
        results[sampling]["Naive Bayes"][metric].append(metrics.accuracy_score(Y_test, clf_bayes.predict(X_test)))
        results[sampling]["RBF Kernel SVC"][metric].append(metrics.accuracy_score(Y_test, clf_rbf.predict(X_test)))
        results[sampling]["Random Forest"][metric].append(metrics.accuracy_score(Y_test, clf_forest.predict(X_test)))
        results[sampling]["AdaBoost"][metric].append(metrics.accuracy_score(Y_test, clf_boost.predict(X_test)))
        results[sampling]["MLP"][metric].append(metrics.accuracy_score(Y_test, clf_mlp.predict(X_test)))

        confusion_matrix = [["TP", "FN"],
                            ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                metric = confusion_matrix[i][j]
                results[sampling]["KNN"][metric].append(metrics.confusion_matrix(Y_test, clf_knn.predict(X_test))[i][j])
                results[sampling]["ID3 (Underfitting)"][metric].append(metrics.confusion_matrix(Y_test, clf_id3_underfit.predict(X_test))[i][j])
                results[sampling]["ID3"][metric].append(metrics.confusion_matrix(Y_test, clf_id3_overfit.predict(X_test))[i][j])
                results[sampling]["ID3 (Overfitting)"][metric].append(metrics.confusion_matrix(Y_test, clf_id3_overfit.predict(X_test))[i][j])
                results[sampling]["CART"][metric].append(metrics.confusion_matrix(Y_test, clf_cart.predict(X_test))[i][j])
                results[sampling]["Naive Bayes"][metric].append(metrics.confusion_matrix(Y_test, clf_bayes.predict(X_test))[i][j])
                results[sampling]["RBF Kernel SVC"][metric].append(metrics.confusion_matrix(Y_test, clf_rbf.predict(X_test))[i][j])
                results[sampling]["Random Forest"][metric].append(metrics.confusion_matrix(Y_test, clf_forest.predict(X_test))[i][j])
                results[sampling]["AdaBoost"][metric].append(metrics.confusion_matrix(Y_test, clf_boost.predict(X_test))[i][j])
                results[sampling]["MLP"][metric].append(metrics.confusion_matrix(Y_test, clf_mlp.predict(X_test))[i][j])

        for sample_key, sampling_dict in results.items():
            for model_key, model_dict in sampling_dict.items():
                for metric_key, trial_results in model_dict.items():
                    print(f'{sample_key} {model_key} {metric_key}: {trial_results}')
        
        metric = "precision"
        for model in models:
            print(results[sampling][model]["TP"])
            print(results[sampling][model]["TP"][-1])
            val = results[sampling][model]["TP"][-1] / (results[sampling][model]["TP"][-1]+results[sampling][model]["FP"][-1])
            print(val)
            results[sampling][model][metric].append(val)

        metric = "recall"
        for model in models:
            results[sampling][model][metric].append(results[sampling][model]["TP"][-1] / (results[sampling][model]["TP"][-1]+results[sampling][model]["FN"][-1]))

        metric = "f1"
        for model in models:
            results[sampling][model][metric].append(2*results[sampling][model]["precision"][-1]*results[sampling][model]["recall"][-1] / (results[sampling][model]["precision"][-1]+results[sampling][model]["recall"][-1]))
        
    new_now = datetime.now()
    elapsed = new_now - now
    print(f'trial {trial} ran in {elapsed} time')
    now = new_now

# flatten results to a 2D array
data = [ ["Sampling", "Model", "Metric", "Average", "Standard Deviation"] + [f'Trial {i}' for i in range(num_trials)] ]
for sampling, sampling_dict in results.items():
    for model, model_dict in sampling_dict.items():
        for metric, trial_results in model_dict.items():
            data.append([sampling, model, metric, statistics.mean(trial_results), statistics.stdev(trial_results)] + trial_results)

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

with open(f'./output/output-{dt_string}.csv',"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(data)