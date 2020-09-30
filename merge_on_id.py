#!/usr/local/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

features_df = pd.read_csv('./data/integrated_features.csv', sep=",", engine="python")
print(features_df)

labels_df = pd.read_csv('./data/integrated_labels.csv', sep=",", engine="python")
print(labels_df)

mergedDf = features_df.merge(labels_df, left_on='ID', right_on='ID')
print(mergedDf)

mergedDf.to_csv('./data/integrated_features_and_labels.csv', index=False)

# join by headers
'''cols_dict = dict((header, []) for header in headers)
for df in frames:
  n, m = df.shape
  for header in headers:
    print(header)
    col_from_df = df.get(header, [' '] * n)
    print(col_from_df)
    cols_dict[header].extend(col_from_df)
  print("nice!")
  #df_sum = pd.DataFrame.from_dict(cols_dict)
print("really nice!")
df_sum = pd.DataFrame.from_dict(cols_dict)
print(df_sum)

# print()
# print(df['party'].value_counts().to_string())
# drop the scatter rows
# df['party'].value_counts().plot(kind='bar')
years = [1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996, 1998]
states = set(df['state'].to_list())

for state in states:
  for year in years:
    temp_df = df.loc[(df['state'] == state) & (df['year'] == year)]
    max_num_districts = temp_df['district'].max()
    for district in range(1, max_num_districts+1):
      temp_district_df = temp_df.loc[df['district'] == district]
      to_append_df = temp_district_df[temp_district_df['candidatevotes'] == temp_district_df['candidatevotes'].max()]
      new_df = new_df.append(to_append_df)

df = new_df.sort_values(by=['year', 'state', 'district'])
df_sum.to_csv('integrated_features.csv', index=False)'''
