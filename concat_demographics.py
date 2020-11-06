#!/usr/local/bin/python3
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import math

start_year = 78
all_years = pd.DataFrame()
ids = []
for year in range(start_year, 100, 2):
  df = pd.read_csv(f'./data/congressional_demographic_data/fin{year}.csv', sep=",", engine="python")
  for index, row in df.iterrows():
    if row['CD'] == 0 or row['CD'] == 99 and row['STATE'] != "":
      continue
    district_id = f"19{year}-{row['STATE']}-{row['CD']}"
    ids.append(district_id)
    all_years = all_years.append(row)
  print(all_years)
all_years['ID'] = ids
all_years.to_csv('./data/flipped_label_data/concat_demographic_features.csv')
  
# frames = []
# start_year = 1976
# for file_num in range(95,106):
#   year = (file_num-95)*2 + start_year
#   df = pd.read_csv(f'~/Downloads/fin{file_num}.csv', sep=",", engine="python")
#   n, m = df.shape
#   year_list = [year] * n
#   df.insert(0, "YEAR", year_list, True)
#   #print(df.columns.values.tolist())
#   frames.append(df)


# headers = ["YEAR","STATE","FIPSTATE","SC","CD","REDIST","ACEREGIO","AGE65",
#   "BANK","BLACK","BLUCLLR","CITY","COAST","CONSTRUC","CVLLBRFR","DC","ENROLL",
#   "FARMER","FEDWRKR","FINANCE","FLOOD","FORBORN","GVTWRKR","INTRLAND","LANDSQMI",
#   "LOCLWRKR","MANUF","MARINE","MDNINCM","MILTINST","MILTMAJR","MILTPOP",
#   "NUCPLANT","POPSQMI","POPULATN","PORT","RURLFARM","STATWRKR","TRANSPRT",
#   "UNEMPLYD","UNION","URBAN","VABEDS","VETERANS","WHLRETL"]

# # join by headers
# cols_dict = dict((header, []) for header in headers)
# for df in frames:
#   n, m = df.shape
#   for header in headers:
#     print(header)
#     col_from_df = df.get(header, [' '] * n)
#     print(col_from_df)
#     cols_dict[header].extend(col_from_df)
#   print("nice!")
#   #df_sum = pd.DataFrame.from_dict(cols_dict)
# print("really nice!")
# df_sum = pd.DataFrame.from_dict(cols_dict)
# print(df_sum)

# # print()
# # print(df['party'].value_counts().to_string())
# # drop the scatter rows
# # df['party'].value_counts().plot(kind='bar')
# '''years = [1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996, 1998]
# states = set(df['state'].to_list())

# for state in states:
#   for year in years:
#     temp_df = df.loc[(df['state'] == state) & (df['year'] == year)]
#     max_num_districts = temp_df['district'].max()
#     for district in range(1, max_num_districts+1):
#       temp_district_df = temp_df.loc[df['district'] == district]
#       to_append_df = temp_district_df[temp_district_df['candidatevotes'] == temp_district_df['candidatevotes'].max()]
#       new_df = new_df.append(to_append_df)

# df = new_df.sort_values(by=['year', 'state', 'district'])'''
# df_sum.to_csv('integrated_features.csv', index=False)
