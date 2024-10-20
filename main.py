# main.py

import os
from dj_mixer import mix_tracks_enhanced
def main():
    print("Welcome to the Enhanced AI DJ App!")
    print("Please place your song files in the 'songs' directory.")
    song_files = os.listdir('songs')
    if len(song_files) < 2:
        print("You need at least two songs to create a mix.")
        return

    print("Available songs:")
    for idx, song in enumerate(song_files):
        print(f"{idx + 1}: {song}")

    # Select songs
    song1_idx = int(input("Select the first song by number: ")) - 1
    song2_idx = int(input("Select the second song by number: ")) - 1

    song1_path = os.path.join('songs', song_files[song1_idx])
    song2_path = os.path.join('songs', song_files[song2_idx])

    # Mix songs
    output_path = 'mixed_song_enhanced.wav'
    success = mix_tracks_enhanced(song1_path, song2_path, output_path)

    if success:
        print(f"Songs mixed successfully! Output saved to {output_path}")
    else:
        print("An error occurred during mixing.")

if __name__ == "__main__":
    main()
