# Image Caption Generator using CNN & NLP

## Overview
This project implements an Image Caption Generator using a CNN (Convolutional Neural Network) for feature extraction and LSTM (Long Short-Term Memory) for text generation. The model is trained on the Kaggle Flickr dataset to generate meaningful captions for images.

## Workflow

### 1. Data Collection & Preprocessing
**Dataset Used**: Kaggle Flickr Image Captioning Dataset

#### Data Preprocessing Steps:
- Loaded images and captions from the dataset
- Tokenized and cleaned text data (removing punctuations, converting to lowercase, etc.)
- Mapped images to their respective captions
- Resized and normalized images for CNN input
- Converted text data to sequences using Tokenizer
- Created padded sequences for consistent input size

### 2. Feature Extraction using CNN (Pretrained Model)
- Used InceptionV3 (or ResNet50) as a feature extractor
- Removed the last fully connected layer and extracted feature vectors
- Stored extracted features in a dictionary for faster processing

### 3. Text Processing using LSTM
- Embedded words into vector space using Word Embeddings (GloVe/Word2Vec)
- Designed an LSTM-based language model to generate captions
- **Input**: Extracted image features + processed text sequences
- **Output**: Predicted next word in the caption

### 4. Model Training
- Used Categorical Cross-Entropy Loss for training
- Optimized using Adam optimizer
- Monitored training loss and adjusted hyperparameters (learning rate, batch size, etc.)
- Performed teacher forcing for efficient training

### 5. Caption Generation & Inference
- Processed a test image through the trained CNN-LSTM pipeline
- Generated captions by predicting one word at a time
- Used Beam Search Decoding for improved caption generation

### 6. Evaluation Metrics
- Evaluated the model using BLEU scores to assess caption quality

### 7. Deployment & Future Improvements
- Deployed as a simple Flask Web App for user interaction
- **Future improvements include**:
  - Using Transformer models (e.g., ViT + GPT) for better captioning
  - Implementing attention mechanisms for better context handling

## Installation & Usage

### Prerequisites
Install the required libraries:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/venu284/Image-Caption-Generator-using-CNN-and-NLP/blob/main/notebooks/Image_Caption_Generator.ipynb)



```bash
pip install tensorflow numpy pandas matplotlib nltk keras tqdm./



