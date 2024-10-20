# audio_utils.py

import numpy as np
import librosa
from scipy.signal import butter, lfilter

def load_audio(file_path, sr=None):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def save_audio(file_path, y, sr):
    sf.write(file_path, y, sr)

def normalize_gain(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y**2))
    gain = target_rms / (rms + 1e-6)  # Add epsilon to avoid division by zero
    y_normalized = y * gain
    return y_normalized

def apply_eq(y, sr, gains):
    # gains is a dictionary with 'low', 'mid', 'high' keys
    low_cutoff = 200  # Hz
    high_cutoff = 5000  # Hz

    # Low frequencies
    b, a = butter(2, low_cutoff / (sr / 2), btype='low')
    low = lfilter(b, a, y) * gains.get('low', 1.0)

    # Mid frequencies
    b, a = butter(2, [low_cutoff / (sr / 2), high_cutoff / (sr / 2)], btype='band')
    mid = lfilter(b, a, y) * gains.get('mid', 1.0)

    # High frequencies
    b, a = butter(2, high_cutoff / (sr / 2), btype='high')
    high = lfilter(b, a, y) * gains.get('high', 1.0)

    y_eq = low + mid + high
    return y_eq

def apply_filter(y, sr, cutoff_freq, filter_type='low'):
    b, a = butter(2, cutoff_freq / (sr / 2), btype=filter_type)
    y_filtered = lfilter(b, a, y)
    return y_filtered

def adjust_key(y, sr, n_steps):
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y_shifted

def create_loop(y, sr, start_time, end_time, repeat=2):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    loop_section = y[start_sample:end_sample]
    y_looped = np.tile(loop_section, repeat)
    return y_looped

def apply_scratching(y, sr, scratch_times):
    y_scratch = np.copy(y)
    for start_time, duration in scratch_times:
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        y_scratch[start_sample:end_sample] = y_scratch[start_sample:end_sample][::-1]
    return y_scratch
