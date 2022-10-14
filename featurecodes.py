"""
Author : Beholder
Date : 2022.10.10
"""
import joblib
import numpy as np
import pandas as pd


def tocode(dic):
    return list(dic.values())


initial = pd.read_csv('datatxt.csv')
df = joblib.load('features.JL')
df_encoded = pd.DataFrame(data=None, index=df.index, columns=df.columns)
df_encoded['label'] = initial['label'].values
for col in df.columns:
    df_encoded[col] = df[col].apply(tocode)
for col in df_encoded.columns[:-1]:
    length = len(max(df_encoded[col], key=len))
    for index in df_encoded.index:
        df_encoded[col][index] = df_encoded[col][index] + [0.0]*(length-len(df_encoded[col][index]))
for col in df_encoded.columns[:-1]:
    df_encoded[col] = df_encoded[col].apply(np.array, dtype='float')
joblib.dump(df_encoded, 'featurecodes.JL')
print(df_encoded['AAC'].values)