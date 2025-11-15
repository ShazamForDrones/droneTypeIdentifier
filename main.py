import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

import librosa.feature
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import datetime

# Fixe le random seed pour reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

drone_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


# ============ DATA AUGMENTATION TOUJOURS (pas aléatoire) ============
def augment_audio(mel_db):
    """Augmente TOUJOURS - pas de random!"""
    augmented = [mel_db]  # Original
    
    # Bruit léger (TOUJOURS)
    noise = np.random.normal(0, 0.005, mel_db.shape)
    augmented.append(mel_db + noise)
    
    # Time shift (TOUJOURS)
    shift = np.random.randint(-40, 40)
    augmented.append(np.roll(mel_db, shift, axis=1))
    
    # Pitch shift (TOUJOURS)
    pitch = np.random.uniform(0.95, 1.05)
    augmented.append(mel_db * pitch)
    
    return augmented  # TOUJOURS 4 versions (1 original + 3 augmentées)


"""
conventions du ML X = input, y = output
X est l'array qui represente les enregistrements audio des drones
y est le type de drone prédit
"""
X, y = [], []

print("Chargement des données...")
for drone_type in drone_types:
    folder = f"Training_data/{drone_type}"
    for filename in os.listdir(folder):
        audio_path = os.path.join(folder, filename)

        y_audio, sr = librosa.load(audio_path, duration=30)
        # creation spectrogramme
        mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        # Convertie en decibel
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # pour consitent data
        max_len = 660
        if mel_db.shape[1] < max_len:
            pad_width = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]

        augmented_samples = augment_audio(mel_db)
        for aug_mel in augmented_samples:
            X.append(aug_mel)
            y.append(drone_types.index(drone_type))

X = np.array(X)
y = to_categorical(np.array(y))

print(f"Dataset total: {len(X)} échantillons (1000 originaux × 3 = 3000)")

# Normaliser les données entre 0 et 1 (important!)
X = (X - X.min()) / (X.max() - X.min())

X = X[..., np.newaxis] # ajout d'un 4e axe
# 20% des data va dans test le reste on train avec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


model = Sequential([
    Input(shape=(128, 660, 1)),

    # 3 Conv layers seulement (pas 4!)
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),  # Augmenté 0.4 → 0.5

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    # Retiré Conv2D(256) - trop complexe!

    Flatten(),
    
    # 1 Dense seulement (pas 2!)
    Dense(128, activation='relu'),
    Dropout(0.7),  # Augmenté 0.6 → 0.7

    Dense(len(drone_types), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Nombre d'échantillons:", len(X_train), "train,", len(X_test), "test")
print("Shape X_train:", X_train.shape)
print("Shape y_train:", y_train.shape)
print(f"Batches par epoch: {len(X_train) // 32}")

# stop quand le model n'apprend plus
# patience = nombre d'epochs à attendre sans amélioration avant d'arrêter
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n{'='*60}")
print(f"TEST ACCURACY: {test_acc*100:.2f}%")
print(f"{'='*60}")

# FIX: Remplace : par - dans le nom du fichier
model.save(f'models/{datetime.datetime.now().strftime("%m-%d_%H-%M-%S")}.keras')


# graph
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""
visualisation

X = np.array(X)
print("X shape:", X.shape)

mel_db = X[0]
label_index = y[0]
drone_type_name = drone_types[label_index]

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=22050, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title(f"Drone Type: {drone_type_name}")
plt.tight_layout()
plt.show()
"""

