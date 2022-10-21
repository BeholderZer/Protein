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
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

ClassifierDict = {}
ClassifierDict['ET'] = ExtraTreesClassifier(random_state=0)
ClassifierDict['NB'] = GaussianNB()
ClassifierDict['KNN'] = KNeighborsClassifier(n_neighbors=5, weights='distance')
ClassifierDict['LR'] = LogisticRegression(multi_class='multinomial', max_iter=10000, random_state=0)
ClassifierDict['RF'] = RandomForestClassifier(random_state=0)
ClassifierDict['SVM'] = SVC(C=0.5, kernel='rbf', decision_function_shape='ovr')


def protein_type(s):
    it = {b'Inner_Membrane': 0, b'Intermembrane_Space': 1, b'Matrix': 2, b'Outer_Membrane': 3}
    return it[s]


def RunMoldel(col, clfname, x, y):
    res = []
    sfolder = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in sfolder.split(x, y):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        clf = ClassifierDict[clfname]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        res.append([GetAcc(y_test, y_pred), GetSn(y_test, y_pred), GetSp(y_test, y_pred), GetMcc(y_test, y_pred),
                    GetGcc(y_test, y_pred)])
    row = [col, clfname] + [*np.array(res).mean(axis=0)]
    return row


def GetGcc(y_true, y_pred, k=4):
    temp = 0.0
    h = len(y_true)
    c_matrix = confusion_matrix(y_true, y_pred)
    m = np.sum(c_matrix, axis=1)
    n = np.sum(c_matrix, axis=0)
    for i in range(k):
        for j in range(k):
            t = m[i] * n[j] / h
            if t == 0:
                break
            temp += (c_matrix[i][j] - t) ** 2 / t
    gcc = (temp / h * (k - 1)) ** 0.5

    return gcc


def GetSn(y_true, y_pred, k=4):
    sen = []
    con_mat = confusion_matrix(y_true, y_pred)
    for i in range(k):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return np.array(sen).mean()


def GetSp(y_true, y_pred, k=4):
    spe = []
    con_mat = confusion_matrix(y_true, y_pred)
    for i in range(k):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return np.array(spe).mean()


def GetAcc(y_true, y_pred, k=4):
    acc = []
    con_mat = confusion_matrix(y_true, y_pred)
    for i in range(k):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)

    return np.array(acc).mean()


def GetMcc(y_true, y_pred, k=4):
    mcc = []
    con_mat = confusion_matrix(y_true, y_pred)
    for i in range(k):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        under = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        if under == 0:
            mcc1 = (tp * tn - fp * fn)
        else:
            mcc1 = (tp * tn - fp * fn) / under
        mcc.append(mcc1)

    return np.array(mcc).mean()


res = []
columns = ['features', 'Classifier', 'Acc', 'Sn', 'Sp', 'Mcc', 'Gcc']
df = joblib.load('featurecodes.JL')
tag = np.loadtxt('datatxt.csv', dtype=object, delimiter=',', converters={2: protein_type}, skiprows=1)[:, 2]
for col in df.columns[:-1]:
    X = np.array([x for i in df[col].values for x in i], dtype=float).reshape(424, -1)
    y = np.array(tag, dtype=int)
    for key in ClassifierDict:
        res.append(RunMoldel(col, key, X, y))
df = pd.DataFrame(res, columns=columns)
df.to_csv('res.csv')
print(df)
