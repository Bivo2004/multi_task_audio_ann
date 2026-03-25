import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_PATH = os.path.join(BASE_DIR, "models", "learning_curves.png")

# Data extracted exactly from your heavy-model training logs!
epochs = list(range(1, 26))
train_loss = [3.7166, 3.0414, 2.5446, 2.1333, 1.8278, 1.5223, 1.0851, 0.7806, 0.5533, 0.3350, 
              0.1936, 0.1245, 0.0689, 0.0469, 0.0371, 0.0197, 0.0123, 0.0092, 0.0082, 0.0066, 
              0.0078, 0.0060, 0.0047, 0.0049, 0.0042]

val_loss = [4.2432, 2.7696, 3.5625, 2.8224, 2.7047, 2.2972, 2.3003, 2.2897, 2.5058, 2.1982, 
            2.2095, 2.0249, 2.0356, 2.2741, 2.0152, 1.9324, 1.8645, 1.9208, 1.8738, 1.7850, 
            1.9286, 1.8907, 1.9280, 1.9655, 1.9739]

def plot_learning_curves():
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2, color='#2ca02c')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linewidth=2, color='#d62728')
    
    # Highlight the best epoch (Epoch 20)
    best_epoch = 20
    plt.axvline(x=best_epoch, color='blue', linestyle='--', alpha=0.5, label=f'Best Model (Epoch {best_epoch})')
    plt.scatter(best_epoch, 1.7850, color='blue', s=100, zorder=5)
    
    plt.title('Multi-Task Audio ANN: Learning Curves (Heavyweight CNN)', fontsize=16, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (CrossEntropy)', fontsize=12)
    plt.xticks(epochs[::2]) # Show every other epoch on x-axis
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"✅ Learning curve chart saved to: {SAVE_PATH}")
    
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()