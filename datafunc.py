"""
Author : Beholder
Date : 2022.10.10
"""
from propy import PyPro
import pandas as pd
import joblib

# dictionary
lamda = 10
weight = 0.5
Func_dict = {}
Func_dict['AAC'] = lambda m: PyPro.CalculateAAComposition(m)
Func_dict['DPC'] = lambda m: PyPro.CalculateDipeptideComposition(m)
Func_dict['MBauto'] = lambda m: PyPro.CalculateNormalizedMoreauBrotoAutoTotal(m)
Func_dict['Moranauto'] = lambda m: PyPro.CalculateMoranAutoTotal(m)
Func_dict['Gearyauto'] = lambda m: PyPro.CalculateGearyAutoTotal(m)
Func_dict['CTD'] = lambda m: PyPro.CalculateCTD(m)
Func_dict['SOCN'] = lambda m: PyPro.GetSequenceOrderCouplingNumberTotal(m)
Func_dict['QSO'] = lambda m: PyPro.GetQuasiSequenceOrder(m)
Func_dict['PAAC'] = lambda m: PyPro._GetPseudoAAC(m, lamda=lamda, weight=weight)
Func_dict['APAAC'] = lambda m: PyPro.GetAPseudoAAC(m, lamda=lamda, weight=weight)


def EncoderByProtein(seqprotein):
    traits = []
    for i in Func_dict:
        traits.append(Func_dict[i](seqprotein))
    return traits


columns = ['AAC', 'DPC', 'MBauto', 'Moranauto', 'Gearyauto', 'CTD', 'SOCN', 'QSO', 'PAAC', 'APAAC']
protein_df = pd.read_csv('datatxt.csv')
proteins_arr = protein_df['protein'].values
proteins_before = []
for seq in proteins_arr:
    proteins_before.append(EncoderByProtein(seq))
df = pd.DataFrame(proteins_before, index=proteins_arr, columns=columns)
joblib.dump(df, 'features.JL')
print(df)
