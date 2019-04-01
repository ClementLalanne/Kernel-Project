import svm
import spectrum
import numpy as np

class WeightedSpectrums():
    """
    A class that allows to combine additively kernels.
    """
    def __init__(self, k_max, Cs, weights):
        """
        Initializes the class. k_max is the maximum k_spectrum kernel to consider.
        Cs is the array of regulizer parametters to use for each kernel. The are 
        averaged.
        weights are the weights to use in the combiaison.
        """
        def kernel(X, Y):
            ret = weights[2] * spectrum.k_spectrum(X, Y, k=2)
            for i in range(3, k_max+1):
                ret += weights[i] * spectrum.k_spectrum(X, Y, k=i)
            return ret
        clf = svm.SVM(kernel=kernel, C=np.mean(Cs))
        self.clf = clf
        
    def fit(self, X, y):
        """
        Fits f(X)=y
        """
        self.clf.fit(X, y)
        
    def predict(self, X):
        """
        Predicts f(X)=y
        """
        return self.clf.predict(X)