"""
Package supports "Linear", Ridge, and Logistics by default.
"""

from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression

def define_Ridge(params):
    return Ridge(**params)

def define_Linear(params):
    return LinearRegression(**params)

def define_Logistic(params):
    return LogisticRegression(**params)