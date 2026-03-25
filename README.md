# 🎵 Multi-Task Audio ANN: Genre & Emotion Predictor

[![Hugging Face Spaces](https://img.shields.io/badge/🤗_Hugging_Face-Live_Demo-FFD21E)](https://huggingface.co/spaces/Bevnoty/Multi-Task-Audio-Analyzer)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff6b6b)](#)

A production-ready Deep Learning pipeline that simultaneously predicts **Music Genres** and **Speech Emotion Sentiment** from raw audio files. Built from scratch using PyTorch and deployed as a dynamic web app.

## 🧠 Project Overview

Unlike standard classification models, this project utilizes a **Multi-Task Learning** architecture. A single heavy-weight Convolutional Neural Network (CNN) extracts Mel-Frequency Cepstral Coefficients (MFCCs), processes the audio through a shared deep-feature backbone, and routes the data to two distinct classification heads.

### Key Technical Features:
* **Hardware-Agnostic Deployment:** Automatically toggles between CUDA-accelerated GPU execution and standard CPU execution depending on the host machine.
* **Sliding Window Algorithm:** Dynamically chunks full-length 4-minute songs into 3-second overlapping tensors, analyzing them simultaneously in batches and averaging the probabilities for highly stable predictions.
* **Mixed Precision Training (AMP):** Engineered to utilize `torch.amp.autocast`, maximizing VRAM efficiency and cutting training time in half on NVIDIA Ada Lovelace architectures (RTX 40-Series).
* **Batch Normalization & Deep Layers:** Utilizes a 4-block CNN architecture with deep filter stacks (up to 256 channels) and a 512-neuron shared dense layer to capture complex acoustic representations.

## 📊 Model Performance

The model was trained on a merged dataset mapping **GTZAN** (Music Genres) and **RAVDESS** (Speech Emotions), using a customized loss function (`ignore_index=-1`) to handle disjointed labels across the datasets.

*Please insert your `confusion_matrices.png` and `learning_curves.png` here when uploading to GitHub!*
`![Confusion Matrices](models/confusion_matrices.png)`
`![Learning Curves](models/learning_curves.png)`

## 🚀 Live Demo
You can interact with the live model from any device via Hugging Face Spaces:
**[Launch the Live Web App](https://huggingface.co/spaces/Bevnoty/Multi-Task-Audio-Analyzer)**

## 💻 Local Setup & Execution

If you wish to run the app locally on your own machine (supports both CPU and GPU execution):

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/multi_task_audio_ann.git](https://github.com/YOUR_GITHUB_USERNAME/multi_task_audio_ann.git)
cd multi_task_audio_ann