"""
@Date Created: 16/05/2021

Contains basic maths expressions for helping analysis
"""

import numpy as np

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