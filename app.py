import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from model import MultiTaskAudioNet

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, "models", "multi_task_model.pth")
SR = 22050
DURATION = 3
N_MFCC = 40
SAMPLES_PER_TRACK = SR * DURATION

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# --- Initialize Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskAudioNet().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
else:
    raise FileNotFoundError(f"Could not find model weights at {MODEL_PATH}")

def predict_full_audio(audio_path, task):
    """Safely loads uploaded files, chunks them, and predicts using VRAM-safe mini-batches."""
    if audio_path is None:
        return {"Error - No Audio": 1.0}
        
    try:
        signal, _ = librosa.load(audio_path, sr=SR)
        
        chunks = []
        if len(signal) < SAMPLES_PER_TRACK:
            signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)))
            chunks.append(signal)
        else:
            for i in range(0, len(signal), SAMPLES_PER_TRACK):
                chunk = signal[i:i + SAMPLES_PER_TRACK]
                if len(chunk) == SAMPLES_PER_TRACK:
                    chunks.append(chunk)
                    
        mfccs = []
        for chunk in chunks:
            mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
            mfccs.append(mfcc)
            
        batch_size = 8 
        all_genre_probs = []
        all_emotion_probs = []
        
        with torch.no_grad():
            for i in range(0, len(mfccs), batch_size):
                batch_mfccs = mfccs[i:i + batch_size]
                batch_tensor = torch.tensor(np.array(batch_mfccs), dtype=torch.float32).unsqueeze(1).to(device)
                
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out_genre, out_emotion = model(batch_tensor)
                else:
                    out_genre, out_emotion = model(batch_tensor)
                    
                genre_probs = torch.nn.functional.softmax(out_genre, dim=1).cpu().numpy()
                emotion_probs = torch.nn.functional.softmax(out_emotion, dim=1).cpu().numpy()
                
                all_genre_probs.extend(genre_probs)
                all_emotion_probs.extend(emotion_probs)
                
                del batch_tensor 
                
            avg_genre_probs = np.mean(all_genre_probs, axis=0)
            avg_emotion_probs = np.mean(all_emotion_probs, axis=0)
                
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        if task == "music":
            return {GENRES[i]: float(avg_genre_probs[i]) for i in range(len(GENRES))}
        else:
            return {EMOTIONS[i]: float(avg_emotion_probs[i]) for i in range(len(EMOTIONS))}
            
    except Exception as e:
        print(f"Error processing audio: {e}")
        return {"Error": 1.0}

def predict_genre_only(audio_file):
    return predict_full_audio(audio_file, task="music")

def predict_emotion_only(audio_file):
    return predict_full_audio(audio_file, task="speech")

# --- Build the Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎵 Multi-Task Audio Analyzer")
    gr.Markdown("Welcome! Upload full-length songs or voice notes. The AI will slice the audio into chunks, analyze them all simultaneously, and average the results for maximum accuracy!")
    
    with gr.Tabs():
        with gr.TabItem("🎸 Music Genre Predictor"):
            with gr.Row():
                with gr.Column():
                    # Forced to upload only
                    audio_music = gr.Audio(sources=["upload"], type="filepath", label="Upload a Full Song")
                    btn_music = gr.Button("Analyze Full Song", variant="primary")
                with gr.Column():
                    out_music = gr.Label(num_top_classes=3, label="Predicted Genre")
            btn_music.click(predict_genre_only, inputs=audio_music, outputs=out_music)
            
        with gr.TabItem("🗣️ Speech Emotion Analyzer"):
            with gr.Row():
                with gr.Column():
                    # GUARANTEED FIX: Forced to upload only. Bypasses the browser microphone crash.
                    audio_speech = gr.Audio(sources=["upload"], type="filepath", label="Upload a Voice Note")
                    btn_speech = gr.Button("Analyze Emotion", variant="primary")
                with gr.Column():
                    out_speech = gr.Label(num_top_classes=3, label="Predicted Emotion")
            btn_speech.click(predict_emotion_only, inputs=audio_speech, outputs=out_speech)

if __name__ == "__main__":
    interface.launch()