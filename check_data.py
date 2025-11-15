import os
import librosa
import numpy as np

drone_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

print("Vérification des données:\n")

total_files = 0
for drone_type in drone_types:
    folder = f"Training_data/drone_{drone_type}"
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3'))]
        count = len(files)
        total_files += count
        print(f"Drone {drone_type}: {count} fichiers")
        
        # Check first file
        if files:
            audio_path = os.path.join(folder, files[0])
            y, sr = librosa.load(audio_path, duration=10)
            duration = len(y) / sr
            print(f"  Sample: {files[0]}")
            print(f"  Durée: {duration:.2f}s, Sample rate: {sr}Hz")
            print(f"  Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}")
            
            # Check if audio is silent
            if np.abs(y).max() < 0.01:
                print(f"  ⚠️ ATTENTION: Audio très silencieux!")
            print()
    else:
        print(f"Drone {drone_type}: DOSSIER MANQUANT!")

print(f"\nTotal: {total_files} fichiers")
print(f"Avec augmentation 4x: {total_files * 4} échantillons")
print(f"Train/test split (80/20): ~{int(total_files * 4 * 0.8)} train, ~{int(total_files * 4 * 0.2)} test")
print(f"Échantillons par classe (train): ~{int(total_files * 4 * 0.8 / 10)}")
