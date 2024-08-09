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

def generate_square_wave(array_length, num_periods):
    """
    Generate a square wave with specified array length and number of periods.
    
    :param array_length: Length of the output array.
    :param num_periods: Number of periods in the square wave.
    :return: Numpy array representing the square wave.
    """
    samples_per_period = round(array_length / num_periods)
    
    total_samples = samples_per_period * num_periods
    
    square_wave = np.zeros(total_samples)
    
    half_period = samples_per_period // 2
    
    for i in range(num_periods):
        start_index = i * samples_per_period
        square_wave[start_index:start_index + half_period] = 1
    
    if total_samples > array_length:
        square_wave = square_wave[:array_length]
    elif total_samples < array_length:
        square_wave = np.pad(square_wave, (0, array_length - total_samples), 'constant')
    
    return square_wave

def get_sawtooth_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = signal.sawtooth(2*spacing*pi)
    wave = np.tile(wave, period*2)

    return get_wave(wave, norm, A_min, A_max)

def generate_sawtooth_wave(array_length, num_periods):
    """
    Generate a sawtooth wave with specified array length and number of periods.
    
    :param array_length: Length of the output array.
    :param num_periods: Number of periods in the sawtooth wave.
    :return: Numpy array representing the sawtooth wave.
    """
    samples_per_period = round(array_length / num_periods)
    
    total_samples = samples_per_period * num_periods
    
    single_period = np.linspace(0, 1, samples_per_period, endpoint=False)
    
    sawtooth_wave = np.tile(single_period, num_periods)
    
    if total_samples > array_length:
        sawtooth_wave = sawtooth_wave[:array_length]
    elif total_samples < array_length:
        sawtooth_wave = np.pad(sawtooth_wave, (0, array_length - total_samples), 'constant')
    
    return sawtooth_wave

def get_triangle_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = signal.sawtooth(2*spacing*pi, 0.5)
    wave = np.tile(wave, period*2)

    return get_wave(wave, norm, A_min, A_max)

def get_sin_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = np.sin(2*spacing*pi)
    wave = np.tile(wave, period*2)

    return get_wave(wave, norm, A_min, A_max)

def generate_sine_wave(array_length, num_periods):
    """
    Generate a sine wave with specified array length and number of periods.
    
    :param array_length: Length of the output array.
    :param num_periods: Number of periods in the sine wave.
    :return: Numpy array representing the sine wave.
    """
    t = np.linspace(0, num_periods * 2 * np.pi, array_length)
    
    sine_wave = np.sin(t)
    
    return sine_wave

def get_cos_waves(spacing, period, norm=False, A_min=0, A_max=1):
    wave = np.cos(2*spacing*pi)
    wave = np.tile(wave, period*2)
    return get_wave(wave, norm, A_min, A_max)

def get_npy_data(dpath, norm=False, A_min=0, A_max=1):
    data = np.load(dpath)
    return get_wave(data, norm, A_min, A_max)

