# features.py

import numpy as np

def average(data):
    average = np.mean(data)
    return average

def rms(data):
    rms = np.sqrt(np.mean(data**2))

    return rms

def peak(data):
    peaks = np.max(np.abs(data))

    return peaks

def peak_to_peak(data):
    x_max = np.max(data)
    x_min = np.min(data)
    pk2pk = x_max - x_min

    return pk2pk

def crest_factor(data):
    peaks = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data**2))
    crest_factor = peaks/rms

    return crest_factor