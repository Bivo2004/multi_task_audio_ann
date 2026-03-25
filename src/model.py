import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskAudioNet(nn.Module):
    def __init__(self, num_genres=10, num_emotions=8):
        super(MultiTaskAudioNet, self).__init__()
        
        # --- Shared Heavyweight Backbone ---
        # Input shape: (Batch, 1, 40, 130)
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2) # Output: (32, 20, 65)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2) # Output: (64, 10, 32)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2) # Output: (128, 5, 16)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2) # Output: (256, 2, 8)
        
        # Flattened size: 256 * 2 * 8 = 4096
        self.fc_shared = nn.Linear(4096, 512)
        self.bn_shared = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        
        # --- Head 1: Genre Prediction ---
        self.genre_head = nn.Linear(512, num_genres)
        
        # --- Head 2: Emotion Prediction ---
        self.emotion_head = nn.Linear(512, num_emotions)

    def forward(self, x):
        # Pass through Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Pass through Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Pass through Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Pass through Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1) 
        
        # Pass through Shared Dense Layer
        x = self.fc_shared(x)
        x = self.bn_shared(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Pass through distinct heads
        genre_out = self.genre_head(x)
        emotion_out = self.emotion_head(x)
        
        return genre_out, emotion_out

# Quick test to ensure the math aligns perfectly
if __name__ == "__main__":
    model = MultiTaskAudioNet()
    dummy_input = torch.randn(1, 1, 40, 130) 
    out_genre, out_emotion = model(dummy_input)
    print("Heavyweight Model successfully built!")
    print(f"Genre Output Shape: {out_genre.shape}")
    print(f"Emotion Output Shape: {out_emotion.shape}")