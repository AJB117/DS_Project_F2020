import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math



df = pd.read_csv('./data/integrated.csv', sep=",", engine="python")
print(df)

# get columns year, state_po, district
IDs = []
for index, row in df.iterrows():
  ID = str(row['year']) + "-" + str(row['state_po']) + "-" + str(row['district'])
  IDs.append(ID)
df['ID'] = IDs
print(df)

# look at each race and find out who won last year
prev_party = []
prev_candidatevotes = []
prev_totalvotes = []
prev_winratio = []
flip = []
for index, row in df.iterrows():
  ID = row['ID']
  ID_parts = ID.split('-')
  prev_year = int(ID_parts[0]) - 2
  if prev_year < 1976:
    prev_party.append('')
    prev_candidatevotes.append('')
    prev_totalvotes.append('')
    prev_winratio.append('')
    flip.append('')
    continue
  prev_ID = str(prev_year) + "-" + ID_parts[1] + "-" + ID_parts[2]
  prev_row = df.loc[df['ID'] == prev_ID].reset_index(drop=True)
  print(prev_row)
  if len(prev_row) > 0:
    prev_party.append(prev_row['party'][0])
    prev_candidatevotes.append(prev_row['candidatevotes'][0])
    prev_totalvotes.append(prev_row['totalvotes'][0])
    prev_winratio.append(prev_row['winratio'][0])
    flip.append(0 if row['party'] == prev_row['party'][0] else 1)
  else:
    prev_party.append('')
    prev_candidatevotes.append('')
    prev_totalvotes.append('')
    prev_winratio.append('')
    flip.append('')

df['prev_party'] = prev_party
df['prev_candidatevotes'] = prev_candidatevotes
df['prev_totalvotes'] = prev_totalvotes
df['prev_winratio'] = prev_winratio
df['flip'] = flip

# print to csv
df.to_csv('previous_years.csv', index=False)

# the below is unused
'''features_df = pd.read_csv('./data/integrated_features.csv', sep=",", engine="python")
print(features_df)

labels_df = pd.read_csv('./data/integrated_labels.csv', sep=",", engine="python")
print(labels_df)

mergedDf = features_df.merge(labels_df, left_on='ID', right_on='ID')
print(mergedDf)

mergedDf.to_csv('./data/integrated_features_and_labels.csv', index=False)

cols_dict = dict((header, []) for header in headers)
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
df.to_csv('integrated_features.csv', index=False)'''
