import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

drone_types = ['A', 'B', 'C', 'D', 'E']  # Just first 5 for visualization

print("Analysing spectrograms to check if drones are distinguishable...\n")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Mel Spectrograms: First sample from each drone type', fontsize=16)

for idx, drone_type in enumerate(drone_types):
    folder = f"Training_data/drone_{drone_type}"
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    
    if files:
        audio_path = os.path.join(folder, files[0])
        y, sr = librosa.load(audio_path, duration=10)
        
        # Create mel spectrogram (same as training)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Plot raw spectrogram
        librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, 
                                cmap='viridis', ax=axes[0, idx])
        axes[0, idx].set_title(f'Drone {drone_type} - Raw')
        axes[0, idx].set_xlabel('Time')
        axes[0, idx].set_ylabel('Frequency (Hz)')
        
        # Plot normalized (like in training)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        librosa.display.specshow(mel_norm, x_axis='time', y_axis='mel', sr=sr,
                                cmap='viridis', ax=axes[1, idx])
        axes[1, idx].set_title(f'Drone {drone_type} - Normalized')
        axes[1, idx].set_xlabel('Time')
        
        # Print stats
        print(f"Drone {drone_type}:")
        print(f"  Raw - Min: {mel_db.min():.2f}, Max: {mel_db.max():.2f}, Mean: {mel_db.mean():.2f}")
        print(f"  Norm - Min: {mel_norm.min():.2f}, Max: {mel_norm.max():.2f}, Mean: {mel_norm.mean():.2f}")
        print(f"  Dominant freq band (mean per freq): {np.argmax(mel_db.mean(axis=1))} / 128")
        print()

plt.tight_layout()
plt.savefig('drone_spectrograms_comparison.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: drone_spectrograms_comparison.png")
plt.show()
