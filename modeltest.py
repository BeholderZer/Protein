"""
Author : Beholder
Date : 2022.10.12
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import sklearn.metrics

ClassifierDict = {}
ClassifierDict['ET'] = ExtraTreesClassifier(random_state=0)
ClassifierDict['NB'] = GaussianNB()
ClassifierDict['KNN'] = KNeighborsClassifier(n_neighbors=3)
ClassifierDict['LR'] = LogisticRegression(multi_class='multinomial', max_iter=10000, random_state=0)
ClassifierDict['RF'] = RandomForestClassifier(random_state=0)
ClassifierDict['SVM'] = SVC(C=0.5, kernel='rbf', decision_function_shape='ovr')


def protein_type(s):
    it = {b'Inner_Membrane': 0, b'Intermembrane_Space': 1, b'Matrix': 2, b'Outer_Membrane': 3}
    return it[s]


def GetValues(col, clfname, clf, x, y):
    cv_results = cross_validate(clf, x, y, scoring=['accuracy', 'precision_micro', 'recall_micro', 'f1_micro'], cv=5)
    res = [col, clfname]
    for i in list(cv_results.keys())[2:]:
        res.append(round(cv_results[i].mean(), 3))
    return res


res = []
columns = ['features', 'Classifier', 'accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
df = joblib.load('featurecodes.JL')
tag = np.loadtxt('datatxt.csv', dtype=object, delimiter=',', converters={2: protein_type}, skiprows=1)[:, 2]
for col in df.columns[:-1]:
    X = np.array([x for i in df[col].values for x in i], dtype=float).reshape(424, -1)
    y = np.array(tag, dtype=int)
    for key in ClassifierDict:
        clf = ClassifierDict[key]
        res.append(GetValues(col, key, clf, X, y))
df = pd.DataFrame(res, columns=columns)
df.to_csv('res.csv')
print(df)



