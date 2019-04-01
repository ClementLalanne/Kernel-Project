import pandas as pd
import numpy as np
from copy import deepcopy

def load_X(kind):
    l_to_id = {"T": 0, "G": 1, "C": 2, "A": 3}
    fname = "data/X{}.csv".format(kind)
    DNAs = pd.read_csv(fname).seq
    X = np.array([[l_to_id[l] for l in DNA] for DNA in DNAs])
    return X

def load_Y(kind):
    fname = "data/Y{}.csv".format(kind)
    Y = pd.read_csv(fname).Bound
    return Y

def load_dataset(num, val_size=0.25, random_state=None):
    kind = "tr{}".format(num)
    X_train = load_X(kind)
    Y_train = load_Y(kind)
    kind = "te{}".format(num)
    X_test = load_X(kind)    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
    }

def load_datasets(val_size=0.25, random_state=None):
    dss = {}
    for i in range(3):
        dss[i] = load_dataset(i, val_size, random_state)
    return dss

def train_test(ref_clfs, dss):
    Y = np.array([])
    for i, ds in dss.items():
        print("> Dataset {}".format(i))
        clf = deepcopy(ref_clfs[i])
        clf.fit(ds["X_train"], ds["Y_train"])
        Y = np.concatenate((Y, clf.predict(ds["X_test"])))
    return Y

def write_Y(Y):
    fname = "data/Yte.csv"
    np.savetxt(fname,
               np.int32(np.stack((np.arange(len(Y)), Y), axis=1)),
               fmt='%i', delimiter=",", header="Id,Bound", comments="")
