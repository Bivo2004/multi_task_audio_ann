import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Import our architecture from the model.py file
from model import MultiTaskAudioNet 

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.pt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "multi_task_model.pth")

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3

# --- Dataset Class ---
class MultiTaskDataset(Dataset):
    def __init__(self, data_path):
        self.X, self.y_genre, self.y_emotion = torch.load(data_path)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_genre[idx], self.y_emotion[idx]

# --- Training Loop ---
def train_model():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing training on: {device.type.upper()}")
    
    # 2. Load and Split Data (80% Train, 20% Validation)
    dataset = MultiTaskDataset(DATA_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model, Loss, and Optimizer
    model = MultiTaskAudioNet().to(device)
    
    # The magic parameter: ignore_index=-1
    criterion = nn.CrossEntropyLoss(ignore_index=-1) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # RTX 4050 Optimization: Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    best_val_loss = float('inf')
    
    # 4. The Loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, genres, emotions in progress_bar:
            inputs = inputs.to(device)
            genres = genres.to(device)
            emotions = emotions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with Mixed Precision
            with torch.amp.autocast('cuda'):
                out_genre, out_emotion = model(inputs)
                loss_genre = criterion(out_genre, genres)
                loss_emotion = criterion(out_emotion, emotions)
                
                # Combine the losses
                total_loss = loss_genre + loss_emotion
                
            # Backward pass with Scaler
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += total_loss.item()
            progress_bar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, genres, emotions in val_loader:
                inputs, genres, emotions = inputs.to(device), genres.to(device), emotions.to(device)
                
                with torch.amp.autocast('cuda'):
                    out_genre, out_emotion = model(inputs)
                    loss_genre = criterion(out_genre, genres)
                    loss_emotion = criterion(out_emotion, emotions)
                    running_val_loss += (loss_genre + loss_emotion).item()
                    
        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save the best weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🌟 New best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()