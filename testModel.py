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
    """Preprocess audio file as done during training"""
    # Load audio
    y_audio, sr = librosa.load(audio_path, duration=10)
    
    # Create mel spectrogram
    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Pad/truncate to correct length
    max_len = 215
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    
    # Normalize (same method as training)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    
    # Add necessary dimensions (batch, height, width, channels)
    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    
    return mel_db

def predict_drone(model, audio_path):
    """Predict drone type from audio file"""
    try:
        # Preprocess audio
        mel_db = preprocess_audio(audio_path)
        
        # Make prediction
        prediction = model.predict(mel_db, verbose=0)
        
        # Find class with highest probability
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        
        return drone_types[predicted_class], confidence, prediction[0]
    
    except Exception as e:
        return None, 0, None

def main():
    # Get model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Find most recent model
        models = glob.glob('models/*.keras')
        if not models:
            print("Error: No model found in 'models/' folder")
            return
        model_path = max(models, key=os.path.getctime)
        print(f"Using model: {model_path}\n")
    
    # Load model
    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get test directory
    if len(sys.argv) > 2:
        test_dir = sys.argv[2]
    else:
        test_dir = input("Enter path to directory containing test audio files (default: Test_data): ").strip()
        if not test_dir:
            test_dir = "Test_data"
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3']:
        audio_files.extend(glob.glob(os.path.join(test_dir, '**', ext), recursive=True))
    
    if not audio_files:
        print(f"No audio files found in {test_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files\n")
    print("="*80)
    
    # Test each file
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        filename = os.path.basename(audio_path)
        print(f"[{i}/{len(audio_files)}] Testing: {filename}")
        
        predicted_type, confidence, probabilities = predict_drone(model, audio_path)
        
        if predicted_type:
            results.append({
                'file': filename,
                'predicted': predicted_type,
                'confidence': confidence
            })
            
            print(f"  → Prediction: Drone {predicted_type} (confidence: {confidence:.2f}%)")
            
            # Show top 3 probabilities
            top3_indices = np.argsort(probabilities)[::-1][:3]
            print("  Top 3:")
            for idx in top3_indices:
                print(f"    {drone_types[idx]}: {probabilities[idx]*100:.2f}%")
        else:
            print(f"  ✗ Error processing file")
        
        print("-"*80)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        # Count predictions by type
        from collections import Counter
        predictions_count = Counter([r['predicted'] for r in results])
        
        print(f"\nTotal tested: {len(results)} files")
        print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.2f}%")
        print(f"\nPrediction distribution:")
        for drone_type in drone_types:
            count = predictions_count.get(drone_type, 0)
            percentage = (count / len(results)) * 100
            print(f"  Drone {drone_type}: {count} ({percentage:.1f}%)")
        
        # Save results
        output_file = "test_results.txt"
        with open(output_file, 'w') as f:
            f.write("TEST RESULTS\n")
            f.write("="*80 + "\n\n")
            for r in results:
                f.write(f"{r['file']}: Drone {r['predicted']} ({r['confidence']:.2f}%)\n")
        
        print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
