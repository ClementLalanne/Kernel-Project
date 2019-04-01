import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

class SVM():
    """
    Class to Handle SVMs.
    """
    def __init__(self, kernel='linear', C=1.):
        """
        Initializes an SVM solver with regulizer constant C on the dual box constraint.
        """
        if kernel=='linear':
            kernel = lambda x: x
        self.kernel = kernel
        self.C = C

    def fit(self, X, y, precomputed_kernel_train=None):
        """
        Fits the given data F(X)=y. You can pass a precomputed data kernel K(X, X).
        """
        #Computing the kernel
        self.train_samples = X
        self.train_values = y
        if precomputed_kernel_train==None:
            self.kernel_train = self.kernel(X, X) + np.ones((X.shape[0], X.shape[0]))
        else:
            self.kernel_train = precomputed_kernel_train + np.ones(precomputed_kernel_train.shape)

        ysvm = np.array([-1 if u==0 else 1 for u in y])

        P = matrix(self.kernel_train.astype(np.double))
        q = matrix(-ysvm.astype(np.double))
        Gnp = np.concatenate((np.diag(ysvm), -np.diag(ysvm)), axis=0)
        G = matrix(Gnp.astype(np.double))
        hnp = np.concatenate((self.C*np.ones((X.shape[0])), np.zeros(X.shape[0])), axis=0)
        h = matrix(hnp.astype(np.double))

        sol = solvers.qp(P,q,G,h)
        self.alphaopt = np.array(sol['x'])

    def predict(self, X, precomputed_kernel_train_eval=None):
        """
        Predicts f(X). You can pass the precomputed kernel K(X_eval, X_train).
        """ 
        if precomputed_kernel_train_eval==None:
            self.kernel_train_eval = self.kernel(X, self.train_samples) + np.ones((X.shape[0], self.train_samples.shape[0]))
        else:
            self.kernel_train_eval = precomputed_kernel_train_eval + np.ones(precomputed_kernel_train_eval.shape)

        ret = self.kernel_train_eval.dot(self.alphaopt)
        return (ret>0.).astype(int).reshape(-1)