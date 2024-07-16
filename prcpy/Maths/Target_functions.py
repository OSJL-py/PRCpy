from scipy import signal
import numpy as np
from ..Maths.Maths_functions import normalize_list
import matplotlib.pyplot as plt

pi = np.pi

def get_wave(wave, norm, A_min, A_max):
    if norm:
        return normalize_list(wave, A_min, A_max)
    else:
        return wave

def get_square_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = signal.square(2*spacing*pi)
    wave = np.tile(wave, period*2)

    return get_wave(wave, norm, A_min, A_max)

def get_sawtooth_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = signal.sawtooth(2*spacing*pi)
    wave = np.tile(wave, period*2)

    return get_wave(wave, norm, A_min, A_max)

def get_triangle_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = signal.sawtooth(2*spacing*pi, 0.5)
    wave = np.tile(wave, period*2)

    return get_wave(wave, norm, A_min, A_max)

def get_sin_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = np.sin(2*spacing*pi)
    wave = np.tile(wave, period*2)

    return get_wave(wave, norm, A_min, A_max)

def get_cos_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = np.cos(2*spacing*pi)
    wave = np.tile(wave, period*2)
    return get_wave(wave, norm, A_min, A_max)

def get_npy_data(dpath, norm=False, A_min=0, A_max=1):
    data = np.load(dpath)
    return get_wave(data, norm, A_min, A_max)

