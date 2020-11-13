import pandas as pd
import numpy as np
from statistics import mean
from sklearn import preprocessing

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

data = pd.read_csv('../data/flipped_data/merged_features_and_flipped_labels.csv', sep=",", engine="python")
# removal of columns we just don't have enough data on

def replace_with_mean(column):
  years = [1978, 1980, 1982]
  to_replace_with = []
  for year in years:
    for state in states:
      df = data.loc[(data['YEAR'] == year) & (data['STATE'] == state)]
      mean_84 = data[(data['YEAR'] == 1984) & (data['STATE'] == state)][column]
      if len(mean_84.values) > 0:
        df[column] = [mean(mean_84.values)] * len(df)
        print(df[column])
        data.loc[(data['YEAR'] == year) & (data['STATE'] == state)] = df
  data.to_csv('../data/flipped_data/scaled_merged_features_and_flipped_labels.csv')

data = data.drop(['YEAR'], axis=1)
data['NUCPLANT'] = data['NUCPLANT'].fillna(0)

to_standardize = ['CONSTRCT', 'UNION', 'MDNINCM', 'candidatevotes', 'totalvotes', 'winratio', 'prev_winratio', 'prev_candidatevotes', 'prev_totalvotes']
to_normalize_and_divide_by_population = ['VETERANS', 'STATWRKR', 'FEDWRKR','BLACK', 'BLUCLLR', 'ENROLL', 'FARMER', 'FORBORN', 'GVTWRKR', 'MANUF', 'RURLFARM', 'MILTPOP', 'TRANSPRT', 'UNEMPLYD',
                            'URBAN', 'WHLRETL', 'AGE65', 'CVLLBFRC', 'LOCLWRKR']
to_normalize = ['VABEDS', 'BANK', 'FLOOD', 'MILTMAJR', 'MILTINST', 'PORT']

standardize_second = ['STATWRKR', 'BLUCLLR', 'ENROLL', 'GVTWRKR', 'MANUF', 'TRANSPRT', 'URBAN', 'WHLRETL', 'AGE65', 'CVLLBFRC', 'LOCLWRKR']
normalize_second = ['VETERANS', 'FEDWRKR','BLACK', 'FARMER', 'FORBORN', 'RURLFARM', 'MILTPOP', 'UNEMPLYD']

min_max_scaler = preprocessing.MinMaxScaler()
standard_scaler = preprocessing.StandardScaler()

data[to_standardize] = standard_scaler.fit_transform(data[to_standardize])
data[to_normalize] = min_max_scaler.fit_transform(data[to_normalize])

for feature in standardize_second:
  data[feature] = data[feature]/data['POPULATN']

for feature in normalize_second:
  data[feature] = data[feature]/data['POPULATN']

data[standardize_second] = standard_scaler.fit_transform(data[standardize_second])
data[normalize_second] = min_max_scaler.fit_transform(data[normalize_second])

data[['POPSQMI']] = standard_scaler.fit_transform(data[['POPSQMI']])
data[['POPULATN']] = min_max_scaler.fit_transform(data[['POPULATN']])

data.to_csv('../data/flipped_data/scaled_merged_features_and_flipped_labels.csv')