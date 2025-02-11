'''
calc_binary_metrics

Provides implementation of common metrics for assessing a binary classifier's
hard decisions against true binary labels, including:
* accuracy
* true positive rate and true negative rate (TPR and TNR)

Test Cases for calc_TP_TN_FP_FN 
-------------------------------
>>> N = 8
>>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
>>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
>>> TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
>>> TP
2
>>> TN
3
>>> FP
1
>>> FN
2
>>> np.allclose(TP + TN + FP + FN, N)
True

Test Cases for calc_ACC 
-----------------------
# Verify what happens with empty input
>>> acc = calc_ACC([], [])
>>> print("%.3f" % acc)
0.000

>>> N = 8
>>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
>>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
>>> acc = calc_ACC(ytrue_N, yhat_N)
>>> print("%.3f" % acc)
0.625

Test Cases for calc_TPR 
-----------------------
# Verify what happens with empty input
>>> empty_val = calc_TPR([], [])
>>> print("%.3f" % empty_val)
0.000

>>> N = 8
>>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
>>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
>>> tpr = calc_TPR(ytrue_N, yhat_N)
>>> print("%.3f" % tpr)
0.500

Test Cases for calc_TNR 
-----------------------
# Verify what happens with empty input
>>> empty_val = calc_TNR([], [])
>>> print("%.3f" % empty_val)
0.000

>>> N = 8
>>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
>>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
>>> tnr = calc_TNR(ytrue_N, yhat_N)
>>> print("%.3f" % tnr)
0.750

'''

import numpy as np


def calc_TP_TN_FP_FN(ytrue_N, yhat_N):
    ''' Count the four possible states of true and predicted binary values.

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    TP : int
        Number of true positives
    TN : int
        Number of true negatives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives
    '''
    # Cast input to integer just to be sure we're getting what's expected
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    yhat_N = np.asarray(yhat_N, dtype=np.int32)

    # TODO fix by calculating the number of true pos, true neg, etc.
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    return None  # TODO fix me


def calc_ACC(ytrue_N, yhat_N):
    ''' Compute the accuracy of provided predicted binary values.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    acc : float
        Accuracy = ratio of number correct over total number of examples
    '''
    # TODO compute accuracy
    # You should *use* your calc_TP_TN_FP_FN function from above
    # Hint: make sure denominator will never be exactly zero
    # by adding a small value like 1e-10
    return None  # TODO fix me


def calc_TPR(ytrue_N, yhat_N):
    ''' Compute the true positive rate of provided predicted binary values.

    Also known as the recall.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    tpr : float
        TPR = ratio of true positives over total labeled positive
    '''
    # TODO compute TPR
    # You should *use* your calc_TP_TN_FP_FN function from above
    # Hint: make sure denominator will never be exactly zero
    # by adding a small value like 1e-10
    return None  # TODO fix me


def calc_TNR(ytrue_N, yhat_N):
    ''' Compute the true negative rate of provided predicted binary values.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    tnr : float
        TNR = ratio of true negatives over total labeled negative
    '''
    # TODO compute TNR
    # You should *use* your calc_TP_TN_FP_FN function from above
    # Hint: make sure denominator will never be exactly zero
    # by adding a small value like 1e-10
    return None  # TODO fix me
