import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from train import MultiTaskDataset
from model import MultiTaskAudioNet

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.pt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "multi_task_model.pth")

# Labels for our charts
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def evaluate_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data and model for visualization...")
    
    # Load entire dataset for a comprehensive visual
    dataset = MultiTaskDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load our best saved model weights (from Epoch 13)
    model = MultiTaskAudioNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    all_genre_preds, all_genre_true = [], []
    all_emotion_preds, all_emotion_true = [], []

    print("Running predictions...")
    with torch.no_grad():
        for inputs, genres, emotions in loader:
            inputs = inputs.to(device)
            
            with torch.amp.autocast('cuda'):
                out_genre, out_emotion = model(inputs)
            
            # Get highest probability predictions
            _, genre_preds = torch.max(out_genre, 1)
            _, emotion_preds = torch.max(out_emotion, 1)
            
            # Filter out the -1 labels (the ones we told the model to ignore)
            for i in range(len(genres)):
                if genres[i].item() != -1:
                    all_genre_true.append(genres[i].item())
                    all_genre_preds.append(genre_preds[i].item())
                    
                if emotions[i].item() != -1:
                    all_emotion_true.append(emotions[i].item())
                    all_emotion_preds.append(emotion_preds[i].item())

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Multi-Task Audio ANN - Confusion Matrices', fontsize=16)

    # Genre Matrix
    cm_genre = confusion_matrix(all_genre_true, all_genre_preds)
    sns.heatmap(cm_genre, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=GENRES, yticklabels=GENRES)
    axes[0].set_title('Music Genre Predictions (GTZAN)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Emotion Matrix
    cm_emotion = confusion_matrix(all_emotion_true, all_emotion_preds)
    sns.heatmap(cm_emotion, annot=True, fmt='d', cmap='Reds', ax=axes[1], 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    axes[1].set_title('Speech Emotion Predictions (RAVDESS)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "models", "confusion_matrices.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Charts saved successfully to: {save_path}")
    
    # Also open it on your screen right now
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()