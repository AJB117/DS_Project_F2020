import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# replace with an election outcome data file
df = pd.read_csv('../data/integrated.csv', sep=",", engine="python")
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
df.to_csv('../data/flipped_labels.csv', index=False)
