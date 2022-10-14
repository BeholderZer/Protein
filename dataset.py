"""
Author : Beholder
Date : 2022.10.10
"""
import numpy as np
import pandas as pd
import re

with open('data.txt') as f:
    all = re.split('>|\\n|\[|\]\\n', f.read())
all_array = np.array(all).reshape(-1, 4)[:, 1:]
all_df = pd.DataFrame(all_array, columns=['ProteinID', 'label', 'protein'])
all_df.index = all_df.index + 1
all_df.to_csv('datatxt.csv')
