"""
@Date Created: 16/05/2021

Contains basic maths expressions for helping analysis
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def convert_db_to_linear(dbs):
    return 10 ** (np.array(dbs) / 10)

def convert_dBm_to_watts(power):
    return 10 ** ((power-30)/10)

def calc_baseline(xs, ys):
    dy_dx = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    m_x = xs * dy_dx
    y0 = ys[0] - dy_dx * xs[0]
    return m_x + y0

def normalize_list(array, y1=0, y2=1):
    min_val = np.min(array)
    max_val = np.max(array)
    norm_list = ((array - min_val) / (max_val - min_val)) * (y2 - y1) + y1
    return norm_list

def get_sample_spacing(df, period=1):
    sample_rate = int(len(df.index) / period)
    spacing = np.linspace(0, 1, sample_rate, endpoint=True)
    return spacing

###### New Functions #########
def cov(X, Y):
    """ Covariance between random variables X and Y

    :param X : samples of random variable X 
    :param Y : samples of random variable Y
    """
    return np.mean(X*Y) - np.mean(X)*np.mean(Y)

def shannon_entropy(p):
    """ Shannon entropy of a random variable using natural units

    :param p : samples from random variable

    :return : shannon entropy in natural base
    """
    return -np.sum(p * np.log(p))

def estimator_capacity(u, X):
    """ Measures the quality of a linear estimator from a multivariate series
        to reconstruct a univariate series u

    :param u : univariate series to reconstruct
    :param X : multivariate series

    :return : Measure depicting the quality of the estimator bound by the
                interval [0.0, 1.0].
    """
    n_train = int(u.shape[0] * 0.75)
    
    u_train = u[:n_train]
    X_train = X[:n_train]

    u_test  = u[n_train:]
    X_test  = X[n_train:]

    print(X_train.shape,u_train.shape)

    estimator = LinearRegression().fit(X_train, u_train) 

    u_pred = estimator.predict(X_test) # estimator reconstruction

    return cov(u_test, u_pred)**2 / (np.var(u) * np.var(u_pred))