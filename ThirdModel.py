import os
import random
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, Dense, Input, GlobalMaxPooling1D, Concatenate, \
    BatchNormalization, MaxPooling1D
# from tensorflow.keras.layers import , Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split

import seaborn as sns

import warnings


warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

TEST_SIZE = 0.15
# MAX_FEATURES = 30000
MAX_FEATURES = 48000
BATCH_SIZE = 32
EPOCHS = 25

# INITIALIZE VARIABLES AND CALL NN_NETS_R0
# MAX_SENT_LEN = 105
MAX_SENT_LEN = 141
EMBEDDINGS_DIM = 168


def plot_tweet_lengths(tweets):
    """
    Plots the average and distribution of tweet lengths from a given dataset of tweets.

    Parameters:
    tweets (list of str): List of tweets

    Returns:
    None
    """
    # Calculate tweet lengths
    tweet_lengths = [len(tweet) for tweet in tweets]

    # Calculate average tweet length
    avg_length = np.mean(tweet_lengths)

    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot the distribution of tweet lengths
    sns.histplot(tweet_lengths, bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Distribution of Tweet Lengths')
    ax[0].set_ylabel('Frequency')

    # Plot the average tweet length
    ax[1].axhline(avg_length, color='r', linestyle='--')
    ax[1].text(0.5, avg_length, f'Average Length: {avg_length:.2f}', color='r', ha='center')
    sns.histplot(tweet_lengths, bins=30, kde=True, ax=ax[1])
    ax[1].set_title('Tweet Lengths with Average Line')
    ax[1].set_xlabel('Tweet Length')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def prepare_data(file_name: str, max_sent_len: int, max_features: int, test_size: float = 0.25, random_state=42):
    """
    Function to prepare text data for neural network training.

    Parameters:
    - file_name: str - Path to the CSV file containing the data.
    - max_sent_len: int - Maximum sentence length for padding.

    Returns:
    - train_sequences: np.ndarray - Padded training sequences.
    - test_sequences: np.ndarray - Padded testing sequences.
    - train_labels: np.ndarray - Training labels.
    - test_labels: np.ndarray - Testing labels.
    """
    print("max_sent_len =", max_sent_len)

    # Reading CSV data
    print("Reading text data for classification and building representations...")
    df = pd.read_csv(file_name, delimiter="\t")

    # Data Preparation
    df = df[df['Sentiment_label'].isin(['Positive', 'Negative'])].reset_index(drop=True)

    label_mapping = {"Positive": 1, "Negative": 0}
    df["Sentiment_label"] = df["Sentiment_label"].map(label_mapping)
    print(df[df['Sentiment_label'] == 0].shape)
    print(df[df['Sentiment_label'] == 1].shape)

    plot_tweet_lengths(df['Tweet_text'])

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state,
                                         stratify=df['Sentiment_label'])



    train_texts = train_df['Tweet_text'].tolist()
    test_texts = test_df['Tweet_text'].tolist()
    train_labels = train_df['Sentiment_label'].values
    test_labels = test_df['Sentiment_label'].values
    print(len(train_texts), len(test_texts))
    tokenizer = text.Tokenizer(num_words=max_features,
                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                               lower=True)
    tokenizer.fit_on_texts(train_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    train_sequences = sequence.pad_sequences(train_sequences, maxlen=max_sent_len)
    test_sequences = sequence.pad_sequences(test_sequences, maxlen=max_sent_len)

    return train_sequences, test_sequences, train_labels, test_labels


def convert_to_tf_datasets(train_sequences: np.ndarray,
                           test_sequences: np.ndarray,
                           train_labels: np.ndarray,
                           test_labels: np.ndarray,
                           batch_size: int):
    """
    Function to convert sequences and labels into TensorFlow datasets.

    Parameters:
    - train_sequences: np.ndarray - Padded training sequences.
    - test_sequences: np.ndarray - Padded testing sequences.
    - train_labels: np.ndarray - Training labels.
    - test_labels: np.ndarray - Testing labels.
    - batch_size: int - Size of the batches.

    Returns:
    - train_dataset: tf.data.Dataset - Training dataset.
    - test_dataset: tf.data.Dataset - Testing dataset.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))

    train_dataset = train_dataset.shuffle(len(train_sequences)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def NN_nets_r0(max_features: int, max_sent_len: int, embeddings_dim: int, ):
    """
    Function to build, train, and evaluate a CNN model for Arabic Sentiment Analysis.

    Parameters:
    - file_name: str - Path to the CSV file containing the data.
    - max_sent_len: int - Maximum sentence length for padding.
    - batch_size: int - Size of the batches for training.
    """

    # CNN Model
    input_layer = Input(shape=(max_sent_len,))
    embedding_layer = Embedding(input_dim=max_features, output_dim=embeddings_dim, trainable=True)(
        input_layer)
    dropout_embedding = Dropout(0.5)(embedding_layer)

    # ==================
    conv1 = Conv1D(
        filters=128, kernel_size=3, activation='relu',
        # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )(dropout_embedding)
    conv1 = Conv1D(
        filters=64, kernel_size=3, activation='relu',
        # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )(conv1)
    # conv1 = Conv1D(
    #     filters=32, kernel_size=3, activation='relu',
    #     # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    # )(conv1)
    pool1 = GlobalMaxPooling1D()(conv1)
    pool1 = BatchNormalization()(pool1)


    # =============================
    conv2 = Conv1D(
        filters=128, kernel_size=3, activation='relu',
        # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )(dropout_embedding)
    conv2 = Conv1D(
        filters=64, kernel_size=3, activation='relu',
        # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )(conv2)
    # conv2 = Conv1D(
    #     filters=32, kernel_size=3, activation='relu',
    #     # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    # )(conv2)
    pool2 = GlobalMaxPooling1D()(conv2)
    pool2 = BatchNormalization()(pool2)


    # =============================
    conv3 = Conv1D(
        filters=128, kernel_size=3, activation='relu',
        # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )(dropout_embedding)
    conv3 = Conv1D(
        filters=64, kernel_size=3, activation='relu',
        # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    )(conv3)
    # conv3 = Conv1D(
    #     filters=32, kernel_size=3, activation='relu',
    #     # kernel_regularizer=tf.keras.regularizers.L2(0.01)
    # )(conv3)
    pool3 = GlobalMaxPooling1D()(conv3)
    pool3 = BatchNormalization()(pool3)

    concat = Concatenate()([pool1, pool2, pool3])
    dense = Dense(64, activation='relu')(concat)

    dropout_layer = Dropout(0.25)(dense)
    dense_layer = Dense(1, activation='sigmoid')(dropout_layer)

    model = Model(inputs=input_layer, outputs=dense_layer)

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
        metrics=['accuracy']
    )
    # model.compile(
    #     loss='binary_crossentropy',
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    #     metrics=['accuracy']
    # )

    tf.keras.utils.plot_model(
        model,
        to_file=f'output/PaperMuna-03/model_{datetime.now():%Y%m%d_%H%M%S}.png',
        show_shapes=False,
        rankdir='TB',
        dpi=300,
        show_layer_activations=True,
        show_trainable=True,
    )
    # Print model summary
    print(model.summary())

    return model


def plot_auc(y_true, y_pred, title='ROC Curve', save=True):
    """
    Plot the AUC (Area Under the ROC Curve) given true labels and predictions.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Target scores, can either be probability estimates or confidence values.

    Returns:
    None
    """
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save:
        plt.savefig('output/PaperMuna-03/' + f"model_{datetime.now():%Y%m%d_%H%M%S}_history.png")
    plt.show()


def plot_performance(history=None, figure_directory='./output', ylim_pad=None):
    if ylim_pad is None:
        ylim_pad = [0, 0]

    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history.history['accuracy']
    y2 = history.history['val_accuracy']

    min_y = min(min(y1), min(y2)) - ylim_pad[0]
    max_y = max(max(y1), max(y2)) + ylim_pad[0]

    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy', fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2)) - ylim_pad[1]
    max_y = max(max(y1), max(y2)) + ylim_pad[1]

    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss', fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory + f"model_{datetime.now():%Y%m%d_%H%M%S}_history.png")

    plt.show()


if __name__ == "__main__":
    # Assuming you have a list of files to iterate over, replace with your actual paths

    file = "input/ArSAS.txt"

    print(f'\nProcessing Dataset: {file}')

    train_sequences, test_sequences, train_labels, test_labels = prepare_data(
        file_name=file,
        max_sent_len=MAX_SENT_LEN,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE
    )

    train_dataset, test_dataset = convert_to_tf_datasets(
        train_sequences, test_sequences,
        train_labels, test_labels,
        batch_size=BATCH_SIZE
    )

    model = NN_nets_r0(
        max_features=MAX_FEATURES,
        max_sent_len=MAX_SENT_LEN,
        embeddings_dim=EMBEDDINGS_DIM,
    )

    # Model callbacks
    callbacks = [
        EarlyStopping(patience=3, monitor='val_loss'),
        ModelCheckpoint(filepath=f'./output/PaperMuna-03/model_{datetime.now():%Y%m%d_%H%M%S}_best_weights.keras',
                        save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1, min_delta=0.00001, ),
    ]

    # Model training
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset,
                        callbacks=callbacks)

    # Evaluate model
    print("Evaluate...")
    _, accuracy = model.evaluate(test_dataset)
    # Predictions and metrics
    y_pred = model.predict(test_dataset)
    y_pred_binary = (y_pred > 0.5).astype('int32')
    test_labels_binary = np.concatenate([y for x, y in test_dataset], axis=0)

    with open(f'./output/PaperMuna-03/model_{datetime.now():%Y%m%d_%H%M%S}_results.txt', 'wt') as file:
        print(f'Test accuracy: {accuracy}\n', )
        file.write(f'Test accuracy: {accuracy}\n', )
        print(f"Confusion Matrix:\n{confusion_matrix(test_labels_binary, y_pred_binary)}\n", )
        file.write(f"Confusion Matrix:\n{confusion_matrix(test_labels_binary, y_pred_binary)}\n", )
        print(f"classification_report:\n{classification_report(test_labels_binary, y_pred_binary)}\n")
        file.write(f"classification_report:\n{classification_report(test_labels_binary, y_pred_binary)}\n")
        file.write(f"\n")
        file.write(f"""
        TEST_SIZE = {TEST_SIZE}
        MAX_FEATURES = {MAX_FEATURES}
        BATCH_SIZE = {BATCH_SIZE}
        EPOCHS = {EPOCHS}
        MAX_SENT_LEN = {MAX_SENT_LEN}
        EMBEDDINGS_DIM = {EMBEDDINGS_DIM}
""")
    plot_auc(test_labels_binary, y_pred_binary, title='ROC Curve')

    plot_performance(
        history=history,
        figure_directory='output/PaperMuna-03/'
    )

    # NN_nets_r0(file_name, max_sent_len, )

# Hyperparameters
# Batch Size: Experiment with different batch sizes. settled on 32 at last as the best size for training
# Test size drop to 15%
# Set the random seed to 42 to make the results regenerate
# MAX_FEATURES increase from 30000 to 48000 based on trial and error
# MAX_SENT_LEN increase from 105 to 141 based on statistical mean length of the tweets
# EMBEDDINGS_DIM decrease from 300 to 168 based on trial and error

# Training Process
# Add visualization to select the best hypers based on statistics
# Increase the training epochs to 25
# TODO: Data Augmentation: Augment your dataset with techniques like back translation, synonym replacement, or random insertion of similar words

# Regularization
# L2 Regularization: Apply L2 regularization to Conv1D and Dense layers. -- FOUND TO BE LESS EFFICIENT
# Batch Normalization: Add Batch Normalization layers to stabilize and possibly improve training speed and performance.

# Loss:
# Change loss to become binary instead of categorical

# Optimizers:
# Change Optimizer to become Adam/Nadam instead of Adagrad
# Change Optimizer Learning rate to start from 0.01 instead of 0.001 -- FOUND TO BE LESS EFFICIENT

# Callbacks:
# Change early stopping to become 3 instead of 5
# Change early stopping to monitor the validation accuracy
# Learning Rate: Adjust the learning rate On Plateau.

# Dropout:
# Adjust the dropout rate to 0.25 at lower stages [before the final layer] instead of 0.5

# Convolutional Layers:
# Experiment with varying the number of filters and kernel sizes to capture different features.
# Add more Conv1D layers or increase the number of filters in each Conv1D layer.

# Pooling Layers:
# Use GlobalMaxPooling1D instead of MaxPooling1D, which might capture more relevant features.
# Experiment with different pooling sizes and strides.

# Dense Layers:
# Add one or more dense layers before the final output layer to learn more complex features.
# Experiment with different activation functions (e.g., ReLU, relu) in the dense layers.
