import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import librosa
import numpy as np
from keras.models import load_model
import glob

drone_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

def preprocess_audio(audio_path):
    """Prétraite un fichier audio comme durant l'entraînement"""
    # Charge l'audio
    y_audio, sr = librosa.load(audio_path, duration=10)
    
    # Crée le spectrogramme mel
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Padding/truncate à la bonne longueur
    max_len = 215
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    
    # Normalise (même méthode que l'entraînement)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    
    # Ajoute les dimensions nécessaires (batch, height, width, channels)
    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    
    return mel_db

def predict_drone(model, audio_path):
    """Prédit le type de drone à partir d'un fichier audio"""
    try:
        # Prétraite l'audio
        mel_db = preprocess_audio(audio_path)
        
        # Fait la prédiction
        prediction = model.predict(mel_db, verbose=0)
        
        # Trouve la classe avec la plus haute probabilité
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        
        return drone_types[predicted_class], confidence, prediction[0]
    
    except Exception as e:
        return None, 0, None

def main():
    # Demande le chemin du modèle
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Trouve le modèle le plus récent
        models = glob.glob('models/*.keras')
        if not models:
            print("Erreur: Aucun modèle trouvé dans le dossier 'models/'")
            return
        model_path = max(models, key=os.path.getctime)
        print(f"Utilisation du modèle: {model_path}\n")
    
    # Charge le modèle
    try:
        model = load_model(model_path)
        print("✓ Modèle chargé avec succès\n")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # Demande le répertoire de test
    if len(sys.argv) > 2:
        test_dir = sys.argv[2]
    else:
        test_dir = input("Entrez le chemin du répertoire contenant les fichiers audio de test (par défaut: Test_data): ").strip()
        if not test_dir:
            test_dir = "Test_data"
    
    # Trouve tous les fichiers audio
    audio_files = []
    for ext in ['*.wav', '*.mp3']:
        audio_files.extend(glob.glob(os.path.join(test_dir, '**', ext), recursive=True))
    
    if not audio_files:
        print(f"Aucun fichier audio trouvé dans {test_dir}")
        return
    
    print(f"Trouvé {len(audio_files)} fichiers audio\n")
    print("="*80)
    
    # Teste chaque fichier
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        filename = os.path.basename(audio_path)
        print(f"[{i}/{len(audio_files)}] Test: {filename}")
        
        predicted_type, confidence, probabilities = predict_drone(model, audio_path)
        
        if predicted_type:
            results.append({
                'file': filename,
                'predicted': predicted_type,
                'confidence': confidence
            })
            
            print(f"  → Prédiction: Drone {predicted_type} (confiance: {confidence:.2f}%)")
            
            # Affiche les top 3 probabilités
            top3_indices = np.argsort(probabilities)[::-1][:3]
            print("  Top 3:")
            for idx in top3_indices:
                print(f"    {drone_types[idx]}: {probabilities[idx]*100:.2f}%")
        else:
            print(f"  ✗ Erreur lors du traitement")
        
        print("-"*80)
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    
    if results:
        # Compte les prédictions par type
        from collections import Counter
        predictions_count = Counter([r['predicted'] for r in results])
        
        print(f"\nTotal testé: {len(results)} fichiers")
        print(f"Confiance moyenne: {np.mean([r['confidence'] for r in results]):.2f}%")
        print(f"\nDistribution des prédictions:")
        for drone_type in drone_types:
            count = predictions_count.get(drone_type, 0)
            percentage = (count / len(results)) * 100
            print(f"  Drone {drone_type}: {count} ({percentage:.1f}%)")
        
        # Sauvegarde les résultats
        output_file = "test_results.txt"
        with open(output_file, 'w') as f:
            f.write("RÉSULTATS DES TESTS\n")
            f.write("="*80 + "\n\n")
            for r in results:
                f.write(f"{r['file']}: Drone {r['predicted']} ({r['confidence']:.2f}%)\n")
        
        print(f"\n✓ Résultats sauvegardés dans: {output_file}")

if __name__ == "__main__":
    main()
