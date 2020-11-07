import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

def find(row, year, column, state):
  if row['ID'].split('-')[0] == str(year) and row['ID'].split('-')[1] == state:
    return row

def get_slope(from_year, to_year, column, data):
  state_dict = dict.fromkeys(states, [])
  print(state_dict)
  for state in states:
    from_data = data[data['ID'].str.startswith(str(from_year)) & data['ID'].str.contains(state)]
    to_data = data[data['ID'].str.startswith(str(to_year)) & data['ID'].str.contains(state)]
    # print(from_data[column].values)
    from_data = list(map(lambda x: int(x), from_data[column].values))
    to_data = list(map(lambda x: int(x), to_data[column].values))
    # print(from_data)
    print(from_data)
    from_mean = mean(from_data)
    to_mean = mean(to_data)
  # for state in states:
  #   for i in range(50):
  #     obj_id = f'{from_year}-{state}-{i}'
  #     # print(obj_id)
  #     from_row = data.loc[data['ID'] == obj_id]
  #     state_dict[state].append(from_row)
  print(from_mean)
  print(to_mean)
  # from_data = []
  # to_data = []
  # slope_dict = {}
  # for state in states:
  #   for index, row in data.iterrows():
  #     if row['ID'].split('-')[0] == str(from_year) and row['ID'].split('-')[1] == state:
  #       from_data.append(row[column])
  #     elif row['ID'].split('-')[0] == str(to_year) and row['ID'].split('-')[1] == state:
  #       to_data.append(row[column])
  
  # dist = to_year/from_year
  # to_mean = mean(to_data)
  # from_mean = mean(from_data)
  # print((to_mean - from_mean)/dist)

data = pd.read_csv('./data/flipped_label_data/flipped_integrated_features_and_labels.csv', sep=",", engine="python")
# data = pd.read_csv('./cleaningscripts/testing.csv', sep=",", engine="python")
# removal of columns we just don't have enough data on

data = data.drop(['FIPSTATE', 'REDIST', 'FINANCE', 'FEDWRKR', 'LOCLWRKR', 'STATWRKR', 'EMPLYD', 'URBNFARM'], axis=1)
data["NUCPLANT"] = data["NUCPLANT"].fillna(0)
data["PORT"] = data["PORT"].fillna(0)



data.to_csv('./cleaningscripts/patrick_cleaning/patricktesting.csv')