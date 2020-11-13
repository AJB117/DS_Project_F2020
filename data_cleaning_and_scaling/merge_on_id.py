#!/usr/local/bin/python3
import pandas as pd
import numpy as np
import math

features_df = pd.read_csv('../data/flipped_data/merged_features.csv', sep=",", engine="python")
print(features_df)

labels_df = pd.read_csv('../data/flipped_data/flipped_labels.csv', sep=",", engine="python")
print(labels_df)

mergedDf = features_df.merge(labels_df, left_on='ID', right_on='ID')
print(mergedDf)

mergedDf = mergedDf.drop(['year', 'state', 'state_po', 'district'], axis=1)
mergedDf.to_csv('../data/flipped_data/merged_features_and_flipped_labels.csv', index=False)

