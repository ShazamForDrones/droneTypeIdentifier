"""
Quick script to estimate train_min and train_max for the existing model
by processing a few sample audio files
"""
import os
import numpy as np
import librosa

SR = 22050
DURATION = 3.0
N_MELS = 64
HOP_LENGTH = 1024
MAX_LEN = 200

def process_audio(path):
    """Process audio file and return mel_db spectrogram"""
    try:
        y_audio, sr = librosa.load(path, duration=DURATION, sr=SR, mono=True)
        
        mel = librosa.feature.melspectrogram(
            y=y_audio,
            sr=sr,
            n_mels=N_MELS,
            n_fft=2048,
            hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Padding / truncation
        if mel_db.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_LEN]
        
        return mel_db
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

# Process all audio files in tests folder
test_folder = "tests"
all_mels = []

if os.path.exists(test_folder):
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            path = os.path.join(test_folder, filename)
            mel_db = process_audio(path)
            if mel_db is not None:
                all_mels.append(mel_db)
                print(f"Processed: {filename}")

if all_mels:
    all_mels = np.array(all_mels)
    estimated_min = all_mels.min()
    estimated_max = all_mels.max()
    
    print("\n" + "="*60)
    print("ESTIMATED NORMALIZATION PARAMETERS")
    print("="*60)
    print(f"Estimated train_min: {estimated_min:.4f}")
    print(f"Estimated train_max: {estimated_max:.4f}")
    print(f"\nBased on {len(all_mels)} audio files")
    print("="*60)
    
    print("\nTo use these values:")
    print("1. Create a file: models/normalization_params_11-15_02-19-25.npy")
    print("2. Run this code:")
    print(f"   import numpy as np")
    print(f"   np.save('models/normalization_params_11-15_02-19-25.npy',")
    print(f"           {{'train_min': {estimated_min:.4f}, 'train_max': {estimated_max:.4f}}})")
else:
    print("\n⚠️  No audio files found in 'tests' folder")
    print("Add some audio files to estimate normalization parameters")
    print("\nTypical values for mel-spectrograms:")
    print("  train_min ≈ -80.0 to -60.0")
    print("  train_max ≈ 0.0")
