import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../data/original_label_data/1976-2018-house2.csv", sep=",", engine="python")

no_party = df['party'].fillna('writein')
df['party'] = no_party

df = df.drop(['office', 'writein', 'special', 'candidate', 'runoff', 'mode', 'unofficial', 'version', 'stage', 'state_fips', 'state_cen', 'state_ic'], 1)
new_df = pd.DataFrame()

final_df = pd.DataFrame()

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

switched_array = []

df = new_df.sort_values(by=['year', 'state', 'district'])
print(df['party'].value_counts().to_string())
le = LabelEncoder()
df['party'] = le.fit_transform(df['party'].values)
print(df)
df.to_csv('integrated.csv', index=False)
