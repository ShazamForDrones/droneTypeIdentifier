import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error

import keras
import librosa
import numpy as np
import time

# =================== CONSTANTES (MUST MATCH TRAINING) ===================
SR = 22050          # sample rate
DURATION = 3.0      # 3 secondes par extrait (MUST BE 3.0 to match training!)
N_MELS = 64         # nombre de mels
HOP_LENGTH = 1024   # hop length for mel spectrogram
MAX_LEN = 200       # longueur temporelle max après mels

genres = ['unknown', 'yes_drone']  # ordre des classes (MUST MATCH TRAINING)

# 1. CHARGER LE MODÈLE ET PARAMÈTRES DE NORMALISATION
print("Chargement du modèle...")
model = keras.saving.load_model(r"models\binary_drone_classifier_11-15_02-19-25.keras")
print("✅ Modèle chargé!")

# Try to load normalization parameters
try:
    norm_params = np.load(r"models\normalization_params_11-15_02-19-25.npy", allow_pickle=True).item()
    TRAIN_MIN = norm_params['train_min']
    TRAIN_MAX = norm_params['train_max']
    print(f"✅ Paramètres de normalisation chargés: min={TRAIN_MIN:.4f}, max={TRAIN_MAX:.4f}")
except FileNotFoundError:
    print("⚠️  Fichier de normalisation introuvable. Utilisation de la normalisation par échantillon.")
    print("   Pour de meilleurs résultats, sauvegardez train_min et train_max lors de l'entraînement.")
    TRAIN_MIN = None
    TRAIN_MAX = None


def analyse(song_path):
    """
    processus complet de l'analyse, prediction et affichage pour une chanson
    :param song_path: le full path d'une chanson
    :return: none
    """
    print(f"\nAnalyse de: {os.path.basename(song_path)}")

    # Si chanson > 2 minutes, commence à 60s
    duration = librosa.get_duration(path=song_path)
    offset = 60 if duration > 120 else 0
    y_audio, sr = librosa.load(song_path, duration=DURATION, offset=offset, sr=SR, mono=True)

    # Melspectrogramme (MUST MATCH TRAINING PARAMETERS)
    mel = librosa.feature.melspectrogram(
        y=y_audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Padding / tronquage sur l'axe temps (MUST MATCH TRAINING)
    if mel_db.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :MAX_LEN]

    # Normalisation (MUST MATCH TRAINING EXACTLY)
    if TRAIN_MIN is not None and TRAIN_MAX is not None:
        # Use saved global normalization parameters from training
        denom = (TRAIN_MAX - TRAIN_MIN + 1e-8)
        mel_db = (mel_db - TRAIN_MIN) / denom
    else:
        # Fallback: per-sample normalization (less accurate but works)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    mel_db = mel_db[np.newaxis, ..., np.newaxis]  # Shape: (1, N_MELS, MAX_LEN, 1)


    prediction = model.predict(mel_db, verbose=0)


    genre_index = np.argmax(prediction)
    genre_name = genres[genre_index]
    confidence = prediction[0][genre_index] * 100

    print("\n" + "="*60)
    print(f"GENRE PRÉDIT: {genre_name.upper()}")
    print(f"CONFIANCE: {confidence:.2f}%")
    print("="*60)

    print("\nProbabilités par genre:")
    for i, genre in enumerate(genres):
        prob = prediction[0][i] * 100
        bar = "█" * int(prob / 2)  # Barre visuelle
        print(f"{genre:12} : {prob:5.2f}% {bar}")
    print("="*60)


# Replace the static for-loop with a live watcher that detects new audio files
TEST_FOLDER = "tests"
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.au'}

def is_audio_file(name):
    return os.path.splitext(name)[1].lower() in AUDIO_EXTS

def wait_until_stable(path, checks=6, delay=0.5):
    """Wait until file size stabilizes (to avoid processing half-copied files)."""
    try:
        last = -1
        for _ in range(checks):
            if not os.path.exists(path):
                return False
            size = os.path.getsize(path)
            if size == last:
                return True
            last = size
            time.sleep(delay)
        return True
    except Exception:
        return False

def watch_and_analyse(folder=TEST_FOLDER, poll_interval=1.0):
    seen = set(f for f in os.listdir(folder) if is_audio_file(f))
    # Optionally analyse existing files on start:
    for f in sorted(seen):
        analyse(os.path.join(folder, f))

    print(f"\nWatching '{folder}' for new audio files. Press Ctrl+C to stop.\n")
    try:
        while True:
            try:
                current = set(f for f in os.listdir(folder) if is_audio_file(f))
            except FileNotFoundError:
                print(f"Folder '{folder}' not found. Retrying...")
                time.sleep(poll_interval)
                continue

            new_files = sorted(current - seen)
            for filename in new_files:
                path = os.path.join(folder, filename)
                print(f"\nDetected new file: {filename} - waiting for copy to finish...")
                if not wait_until_stable(path):
                    print(f"Skipping unstable or removed file: {filename}")
                    seen.add(filename)
                    continue
                analyse(path)
                seen.add(filename)

            # clean up seen (in case files were removed)
            seen &= current
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nWatcher stopped by user.")

# Start watching instead of single-run loop
watch_and_analyse(TEST_FOLDER)

