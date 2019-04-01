import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from hashlib import sha1

def segment(X, size):
    """
    Segments the data in the last dim of X to size size. Return is 
    an array of dim dim(X)+1.
    """
    shape = X.shape[:-1] + (X.shape[-1] - size + 1, size)
    strides = X.strides + (X.strides[-1], )

    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def k_grams(X, k, m=4, ret_inds=False):
    """
    Returns the k-grams of X w.r.t its last dimension.
    """
    N, L = X.shape

    encoding = (m ** np.arange(k, dtype=np.uint64)).astype(np.uint64)

    k_grams_indices = segment(X, size=k).dot(encoding).astype(np.uint64)

    xs = np.repeat(np.arange(N), repeats=k_grams_indices.shape[1]).astype(np.uint64)
    ys = k_grams_indices.reshape(-1).astype(np.uint64)

    inds = np.stack((xs, ys), axis=1).astype(np.uint64)
    inds, counts = np.unique(inds, axis=0, return_counts=True)

    res = sparse.csr_matrix((counts, (inds[:, 0], inds[:, 1])),
                            shape=(N, m**k),
                            dtype=np.uint64)
    if ret_inds:
        return res, np.unique(
            np.stack((ys, xs), axis=1), axis=0, return_counts=True).astype(np.uint64)
    return res

global cache_K_spectrum 
cache_K_spectrum = {}

def k_spectrum(X, Y=None, k=2, m=4):
    """
     Computes the k-spectrum kernel between the sequences from X and Y.
     If Y is None, computes the k-spectrum between sequences from X.
    """
    
    h = (sha1(X).hexdigest(), k)
    if h in cache_K_spectrum:
        return cache_K_spectrum[h]

    k_grams_X = k_grams(X, k=k, m=m)
    k_grams_Y = k_grams_X if Y is None else k_grams(Y, k=k, m=m)

    K = k_grams_X.astype(np.uint64).dot(k_grams_Y.astype(np.uint64).T).toarray().astype(np.uint64) 

    cache_K_spectrum[h] = K
    
    return K