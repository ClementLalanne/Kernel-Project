import numpy as np
import svm
import spectrum
import kernel_combin 
import data_handleing 
from copy import deepcopy

#First load the three datasets
dss = data_handleing.load_datasets(0.25, 42)

#Then load the optimal Cs that were computed using k-fold cross validation
Cs = np.load("OptimalCs.npy")

#Then load the weights that are the val accuracy of eack kernel computed using k-fold cross validation
weights = np.load("weights.npy")

alpha = 6 #Temperature
#The weights are passed in an exponential to sharpen the differences.
weightsexp = np.exp(alpha * weights)

#Define the three classifiers for the three datasets.
ref_clfs = [kernel_combin.WeightedSpectrums(15, Cs[0], weightsexp[0]), kernel_combin.WeightedSpectrums(15, Cs[1], weightsexp[1]), kernel_combin.WeightedSpectrums(15, Cs[0], weightsexp[0])]

#Thain the classifiers and produce the test set.
Y_pred = data_handleing.train_test(ref_clfs, dss)

#Write the predictions
data_handleing.write_Y(Y_pred)