#!/usr/local/bin/python3
import pandas as pd
import numpy as np

data = pd.read_csv('../data/flipped_label_data/flipped_concat_demographic_features.csv', sep=",", engine="python")
# removal of columns we just don't have enough data on
data = data.drop(['FIPSTATE', 'REDIST', 'FINANCE', 'FEDWRKR', 'LOCLWRKR', 'STATWRKR'])
