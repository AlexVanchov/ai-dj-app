# dj_mixer.py

import numpy as np
import librosa
import soundfile as sf
from audio_utils import (
    load_audio, normalize_gain, apply_eq, adjust_key, apply_filter
)

def analyze_song_structure(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bar_length = 4  # Assuming 4 beats per bar
    bars = [beat_times[i:i+bar_length] for i in range(0, len(beat_times), bar_length)]
    phrase_length = 8  # Assuming 8 bars per phrase
    phrases = [bars[i:i+phrase_length] for i in range(0, len(bars), phrase_length)]
    return phrases

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key = chroma_mean.argmax()
    return key  # Returns an integer representing the key

def plan_transition(phrases1, phrases2, key1, key2, duration_song1, duration_song2):
    compatible_keys = [(key1 + interval) % 12 for interval in [0, 2, 5, 7, 9]]
    is_compatible = key2 in compatible_keys

    phrase1 = phrases1[-1] if phrases1 else []
    phrase2 = phrases2[0] if phrases2 else []

    default_start_time_song1 = duration_song1 - 30 if duration_song1 > 30 else duration_song1 / 2
    default_start_time_song2 = 30 if duration_song2 > 30 else duration_song2 / 2

    start_time_song1 = phrase1[0][0] if phrase1 else default_start_time_song1
    start_time_song2 = phrase2[0][0] if phrase2 else default_start_time_song2

    return {
        'start_time_song1': start_time_song1,
        'start_time_song2': start_time_song2,
        'is_compatible': is_compatible
    }

def mix_tracks_enhanced(track1_path, track2_path, output_path):
    try:
        # Load and analyze tracks
        y1, sr1 = load_audio(track1_path)
        y2, sr2 = load_audio(track2_path)

        # Ensure same sample rate
        if sr1 != sr2:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Analyze song structure and key
        phrases1 = analyze_song_structure(y1, sr1)
        phrases2 = analyze_song_structure(y2, sr1)
        key1 = detect_key(y1, sr1)
        key2 = detect_key(y2, sr1)

        # Calculate song durations
        duration_song1 = len(y1) / sr1
        duration_song2 = len(y2) / sr1

        # Plan transitions
        transition_info = plan_transition(phrases1, phrases2, key1, key2, duration_song1, duration_song2)

        # Apply harmonic mixing if keys are not compatible
        if not transition_info['is_compatible']:
            key_diff = (key1 - key2) % 12
            y2 = adjust_key(y2, sr1, n_steps=key_diff)

        # Normalize gains
        y1 = normalize_gain(y1)
        y2 = normalize_gain(y2)

        # Apply EQing
        gains_track1 = {'low': 1.0, 'mid': 0.8, 'high': 0.9}
        gains_track2 = {'low': 0.9, 'mid': 1.0, 'high': 0.8}
        y1_eq = apply_eq(y1, sr1, gains_track1)
        y2_eq = apply_eq(y2, sr1, gains_track2)

        # Create transition segments
        start_sample_song1 = int(transition_info['start_time_song1'] * sr1)
        start_sample_song2 = int(transition_info['start_time_song2'] * sr1)

        # Ensure start_sample_song2 is not less than crossfade_samples
        crossfade_duration = 10  # seconds
        crossfade_samples = int(crossfade_duration * sr1)

        if start_sample_song2 < crossfade_samples:
            print("Adjusting start_sample_song2 to avoid negative indexing.")
            start_sample_song2 = crossfade_samples

        # Prepare segments for mixing
        segment1 = y1_eq[:start_sample_song1]
        segment2 = y2_eq[start_sample_song2:]

        # Ensure there are enough samples for crossfading
        segment1_end = y1_eq[start_sample_song1:start_sample_song1 + crossfade_samples]
        segment2_start = y2_eq[start_sample_song2 - crossfade_samples:start_sample_song2]

        # Check if segments are of the correct length
        if len(segment1_end) != crossfade_samples:
            crossfade_samples = len(segment1_end)
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
        elif len(segment2_start) != crossfade_samples:
            crossfade_samples = len(segment2_start)
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
        else:
            # Create fade-in and fade-out arrays
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)

        # Apply crossfade
        segment1_end *= fade_out
        segment2_start *= fade_in
        crossfaded = segment1_end + segment2_start

        # Combine all segments
        mixed = np.concatenate((segment1, crossfaded, segment2))

        # Apply final effects (optional)
        # Example: Apply a low-pass filter during the transition
        transition_start = len(segment1)
        transition_end = transition_start + crossfade_samples
        mixed[transition_start:transition_end] = apply_filter(
            mixed[transition_start:transition_end], sr1, cutoff_freq=8000, filter_type='low'
        )

        # Normalize final output
        mixed = mixed / np.max(np.abs(mixed) + 1e-6)

        # Save the mixed track
        sf.write(output_path, mixed, sr1)
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        return False
