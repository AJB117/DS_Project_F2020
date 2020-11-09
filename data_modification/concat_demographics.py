#!/usr/local/bin/python3
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import math

def renamer(x):
  if x == "UNEMPLY":
    return "UNEMPLYD"
  if x == "CONSTRUC":
    return "CONSTRCT"
  if x == "MDNINCOM":
    return "MDNINCM"
  if x == "CVLLBRFR":
    return "CVLLBFRC"
  if x == "RRLFRM_2":
    return "RURLFARM"
  if x == "NUCLPANT":
    return "NUCPLANT"
  return x

start_year = 95
all_years = pd.DataFrame()
ids = []
years = []
year = 78
for congress in range(start_year, 106):
  df = pd.read_csv(f'../data/original_feature_data/fin{congress}.csv', sep=",", engine="python")
  df.columns = map(renamer, df.columns)
  df = df.loc[:,~df.columns.duplicated()]
  for index, row in df.iterrows():
    if row['CD'] == 0 or row['CD'] == 99 and row['STATE'] != "":
      continue
    district_id = f"19{year}-{row['STATE']}-{row['CD']}"
    years.append(f"19{year}")
    ids.append(district_id)
    all_years = all_years.loc[:,~all_years.columns.duplicated()]
    all_years = all_years.append(row, ignore_index=True)
  year += 2
  print(all_years)
all_years['ID'] = ids
all_years['YEAR'] = years
all_years = all_years.drop(['WHLSALE', 'REDIST'], axis=1)
all_years.to_csv('../data/flipped_data/merged_features.csv')