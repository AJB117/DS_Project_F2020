import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv("1976-2018-house2.csv", sep=",", engine="python")

no_party = df['party'].fillna('writein')
df['party'] = no_party

df = df.drop(['office', 'writein', 'special', 'candidate', 'runoff', 'mode', 'unofficial', 'version', 'stage', 'state_fips', 'state_cen', 'state_ic'], 1)
# print()
print(df['party'].value_counts().to_string())
# drop the scatter rows
# df['party'].value_counts().plot(kind='bar')
# print(df)
# plt.show()