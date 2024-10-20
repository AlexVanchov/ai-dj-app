# mixer.py

import librosa
import numpy as np
import soundfile as sf

def mix_tracks(track1_path, track2_path, output_path):
    try:
        # Load audio files
        y1, sr1 = librosa.load(track1_path, sr=None)
        y2, sr2 = librosa.load(track2_path, sr=None)

        # Ensure same sample rate
        if sr1 != sr2:
            print("Sample rates are different. Resampling...")
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Analyze beats per minute (BPM)
        tempo1, _ = librosa.beat.beat_track(y=y1, sr=sr1)
        tempo2, _ = librosa.beat.beat_track(y=y2, sr=sr2)

        # Debugging statements
        print(f"Type of tempo1: {type(tempo1)}")
        print(f"Value of tempo1: {tempo1}")
        print(f"Type of tempo2: {type(tempo2)}")
        print(f"Value of tempo2: {tempo2}")

        # Handle tempo as array
        if isinstance(tempo1, np.ndarray):
            tempo1 = tempo1[0]
        if isinstance(tempo2, np.ndarray):
            tempo2 = tempo2[0]

        print(f"Tempo of Track 1: {tempo1:.2f} BPM")
        print(f"Tempo of Track 2: {tempo2:.2f} BPM")

        # Time-stretch second track to match the tempo of the first
        rate = tempo1 / tempo2
        y2_aligned = librosa.effects.time_stretch(y2, rate=rate)

        # Trim tracks to the same length
        min_len = min(len(y1), len(y2_aligned))
        y1 = y1[:min_len]
        y2_aligned = y2_aligned[:min_len]

        # Crossfade parameters
        crossfade_duration = 10  # seconds
        crossfade_samples = crossfade_duration * sr1

        # Ensure crossfade_samples is less than min_len
        crossfade_samples = int(min(crossfade_samples, min_len / 2))

        # Create crossfade window
        fade_in = np.linspace(0, 1, crossfade_samples)
        fade_out = np.linspace(1, 0, crossfade_samples)

        # Apply crossfade
        y1[-crossfade_samples:] *= fade_out
        y2_aligned[:crossfade_samples] *= fade_in

        # Combine tracks
        mixed = np.concatenate((
            y1[:-crossfade_samples],
            y1[-crossfade_samples:] + y2_aligned[:crossfade_samples],
            y2_aligned[crossfade_samples:]
        ))

        # Normalize audio to prevent clipping
        mixed = mixed / np.max(np.abs(mixed))

        # Save the mixed track
        sf.write(output_path, mixed, sr1)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
