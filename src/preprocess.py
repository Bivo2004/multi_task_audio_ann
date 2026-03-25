import os
import librosa
import numpy as np
import torch
from tqdm import tqdm

# --- Configuration ---
# This automatically finds the absolute path of your root project folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed")
SR = 22050       # Standard sample rate
DURATION = 3     # 3 seconds of audio
N_MFCC = 40      # Number of MFCC features to extract
SAMPLES_PER_TRACK = SR * DURATION

# Emotion mapping for RAVDESS (01 = neutral, 02 = calm, etc.)
emotion_map = {
    "01": 0, "02": 1, "03": 2, "04": 3, 
    "05": 4, "06": 5, "07": 6, "08": 7
}

# Genre mapping for GTZAN
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
genre_map = {genre: i for i, genre in enumerate(genres)}

def extract_mfcc(file_path):
    """Loads an audio file and extracts padded/truncated MFCCs."""
    try:
        # Load audio
        signal, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        
        # Pad with zeros if shorter than 3 seconds
        if len(signal) < SAMPLES_PER_TRACK:
            signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
        return torch.tensor(mfcc, dtype=torch.float32)
    except Exception as e:
        return None

def process_datasets():
    features = []
    genre_labels = []
    emotion_labels = []

    print("Processing RAVDESS (Emotions)...")
    ravdess_path = os.path.join(DATA_PATH, "ravdess")
    
    # Auto-detect if audio_speech_actors subfolder exists, otherwise use base
    if os.path.exists(os.path.join(ravdess_path, "audio_speech_actors")):
        ravdess_path = os.path.join(ravdess_path, "audio_speech_actors")

    # os.walk automatically hunts down .wav files no matter how they are nested
    for root, _, files in os.walk(ravdess_path):
        wav_files = [f for f in files if f.endswith(".wav")]
        if not wav_files: continue
        
        for file in tqdm(wav_files, desc=f"RAVDESS: {os.path.basename(root)}"):
            # RAVDESS filename format: 03-01-06-01-02-01-01.wav
            parts = file.split("-")
            if len(parts) > 2:
                emotion = emotion_map.get(parts[2])
                if emotion is not None:
                    mfcc = extract_mfcc(os.path.join(root, file))
                    if mfcc is not None:
                        features.append(mfcc)
                        emotion_labels.append(emotion)
                        genre_labels.append(-1) # -1 means "No Genre Label"

    print("\nProcessing GTZAN (Genres)...")
    gtzan_path = os.path.join(DATA_PATH, "gtzan")
    
    # Auto-detect if genres_original subfolder exists, otherwise use base
    if os.path.exists(os.path.join(gtzan_path, "genres_original")):
        gtzan_path = os.path.join(gtzan_path, "genres_original")

    for root, _, files in os.walk(gtzan_path):
        wav_files = [f for f in files if f.endswith(".wav")]
        if not wav_files: continue
        
        for file in tqdm(wav_files, desc=f"GTZAN: {os.path.basename(root)}"):
            # GTZAN filename format: blues.00000.wav
            genre = file.split(".")[0]
            if genre in genre_map:
                mfcc = extract_mfcc(os.path.join(root, file))
                if mfcc is not None:
                    features.append(mfcc)
                    genre_labels.append(genre_map[genre])
                    emotion_labels.append(-1) # -1 means "No Emotion Label"

    # Stack into final tensors
    X = torch.stack(features)
    # Add a channel dimension (Batch, Channels, Height, Width) for our Neural Network
    X = X.unsqueeze(1) 
    y_genre = torch.tensor(genre_labels, dtype=torch.long)
    y_emotion = torch.tensor(emotion_labels, dtype=torch.long)

    print(f"\nFinal Tensor Shapes - X: {X.shape}, Genres: {y_genre.shape}, Emotions: {y_emotion.shape}")
    
    # Ensure processed directory exists before saving
    os.makedirs(OUT_PATH, exist_ok=True)
    
    # Save to disk
    save_file = os.path.join(OUT_PATH, "processed_data.pt")
    torch.save((X, y_genre, y_emotion), save_file)
    print(f"Successfully saved processed data to: {save_file}")

if __name__ == "__main__":
    process_datasets()