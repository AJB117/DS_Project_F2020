import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("./data/outcome_data/1976-2018-house2.csv", sep=",", engine="python")

no_party = df['party'].fillna('writein')
df['party'] = no_party

df = df.drop(['office', 'writein', 'special', 'candidate', 'runoff', 'mode', 'unofficial', 'version', 'stage', 'state_fips', 'state_cen', 'state_ic'], 1)
new_df = pd.DataFrame()

final_df = pd.DataFrame()
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

switched_array = []

# for state in states:
#   for year in years[1:]:
#     past_df = new_df.loc[(new_df['state'] == state) & (new_df['year'] == year - 2)]
#     switched_df = new_df.loc[(new_df['state'] == state) & (new_df['year'] == year)]

#     max_num_districts = new_df['district'].max()
#     for district in range(1, max_num_districts+1):
#       past_district_df = past_df.loc[df['district'] == district]
#       switched_district_df = switched_df.loc[df['district'] == district]
#       # print("past: ", past_district_df)
#       # print("switched: ", switched_district_df)
#       if not past_district_df.empty and not switched_district_df.empty:
#         # print("past: ", past_district_df['party'].values[0])
#         # print("switched: ", switched_district_df['party'].values[0])
#         # switched_array.append(int(past_df['party'].values[0] == switched_df['party'].values[0]))
#         # final_df.append(switched_district_df)

# df = new_df.sort_values(by=['year', 'state', 'district'])
df = new_df.sort_values(by=['year', 'state', 'district'])
print(df['party'].value_counts().to_string())
le = LabelEncoder()
df['party'] = le.fit_transform(df['party'].values)
print(df)
df.to_csv('integrated.csv', index=False)
