# Sarcasm Detection using TensorFlow

This project implements a **Sarcasm Detection** model using **TensorFlow**. The model classifies sentences as **sarcastic** or **not sarcastic** based on their content. The dataset used consists of news headlines, and the goal is to build a binary classification model.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [File Structure](#file-structure)
- [License](#license)

## Project Overview

This project uses **deep learning** to detect sarcasm in text using a **neural network**. The model uses a **recurrent neural network (RNN)** architecture with an embedding layer for text representation, followed by a dense layer for classification.

The dataset used is a collection of sarcastic and non-sarcastic news headlines in JSON format. The classification is binary: 1 for sarcasm, and 0 for non-sarcasm.

## Setup

Before running the code, ensure you have all the necessary dependencies installed:

```bash
pip install tensorflow matplotlib
```

## Downloading the Dataset

You can download the sarcasm dataset directly using the following command:

```bash
!wget --no-check-certificate \
    https://storage.googleapis.com/learning-datasets/sarcasm.json \
    -O /tmp/sarcasm.json
```
The dataset will be saved as sarcasm.json in the /tmp directory.

## How to Run

### Prepare the Data:
- The dataset is loaded and parsed into sentences and labels (sarcasm or not).
- Sentences are tokenized and padded for consistent input length.

### Model Training:
- The model is trained on the preprocessed text using an embedding layer, a global average pooling layer, and dense layers for classification.

### Make Predictions:
Once trained, the model can predict whether new sentences are sarcastic or not. Use the following code to test predictions:

```python
sentences = ["granny starting to fear spiders in the garden might be real", 
             "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))
```

### Visualizing Training Results:
Accuracy and loss for both training and validation sets are plotted after training.

## Model Architecture

The model architecture consists of:

- **Embedding Layer**: Converts words to fixed-size vectors (with dimensions defined by `embedding_dim`).
- **GlobalAveragePooling1D**: Averages the word embeddings across all words in the sentence.
- **Dense Layer**: A fully connected layer with 24 neurons, using ReLU activation.
- **Output Layer**: A sigmoid-activated layer to output a probability between 0 and 1, determining if the sentence is sarcastic or not.

## Training and Evaluation

- **Epochs**: The model is trained for 30 epochs using the Adam optimizer and binary cross-entropy loss function.
- **Metrics**: Accuracy is used to evaluate the model performance on both the training and testing datasets.

### Example Output:

```plaintext
Sentence 1: Sarcasm detected with probability 0.75
Sentence 2: No sarcasm detected with probability 0.25
```

### Visualizing Training Results:
```python
# Accuracy and loss over epochs
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```

## File Structure:
``` plaintext
├── sarcasm.json            # JSON dataset with headlines
├── model_plot.png          # Diagram of model architecture
├── vecs.tsv                # Word embeddings for visualization
├── meta.tsv                # Metadata for word embeddings
├── README.md               # Project overview
├── sarcasm_detection.py    # Main script for training and prediction
```

## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.



