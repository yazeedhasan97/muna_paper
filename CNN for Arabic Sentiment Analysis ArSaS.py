import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


TEST_SIZE = 0.15
DROPOUT_PROB = [0.5, 0.5]
FILTER_SIZES = [3, 5, 7]
NUM_CLASSES = 2
MAX_FEATURES = 30000
BATCH_SIZE = 32
EPOCHS = 5

# INITIALIZE VARIABLES AND CALL NN_NETS_R0
MAX_SENT_LEN = 105
EMBEDDINGS_DIM = 300
NB_FILTER = EMBEDDINGS_DIM

random_state = 0
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


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


def NN_nets_r0(train_dataset, test_dataset, max_features, max_sent_len: int, epochs: int, embeddings_dim: int,):
    """
    Function to build, train, and evaluate a CNN model for Arabic Sentiment Analysis.

    Parameters:
    - file_name: str - Path to the CSV file containing the data.
    - max_sent_len: int - Maximum sentence length for padding.
    - batch_size: int - Size of the batches for training.
    """
    # Prepare data
    # train_sequences, test_sequences, train_labels, test_labels = prepare_data(file_name, max_sent_len)
    # train_dataset, test_dataset = convert_to_tf_datasets(train_sequences, test_sequences, train_labels, test_labels,
    #                                                      batch_size)

    # CNN Model
    print("Method = CNN for Arabic Sentiment Analysis")
    model_variation = 'CNN-non-static'

    input_layer = Input(shape=(max_sent_len,))
    embedding_layer = Embedding(input_dim=max_features, output_dim=embeddings_dim, trainable=True)(
        input_layer)  # max_features is 30000 and embeddings_dim is 128
    dropout_embedding = Dropout(0.5)(embedding_layer)

    conv_layers = []
    for n_gram in filter_sizes:
        conv_layer = Conv1D(filters=nb_filter, kernel_size=n_gram, activation='relu')(dropout_embedding)
        maxpool_layer = MaxPooling1D(pool_size=max_sent_len - n_gram + 1)(conv_layer)
        flat_layer = Flatten()(maxpool_layer)
        conv_layers.append(flat_layer)

    if len(filter_sizes) > 1:
        concat_layer = tf.keras.layers.concatenate(conv_layers)
    else:
        concat_layer = conv_layers[0]

    dropout_layer = Dropout(0.5)(concat_layer)
    dense_layer = Dense(1, activation='sigmoid')(dropout_layer)

    model = Model(inputs=input_layer, outputs=dense_layer)



    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

    tf.keras.utils.plot_model(
        model,
        to_file='output/PaperMuna-03/model.png',
        show_shapes=False,
        rankdir='TB',
        dpi=300,
        show_layer_activations=True,
        show_trainable=True,
    )
    # Print model summary
    print(model.summary())

    # Model callbacks
    early_stopping = EarlyStopping(patience=20)
    checkpoint_path = f'./output/{os.path.basename(file_name)}_{model_variation}_weights_Ar_best.keras'
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True)

    # Model training
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset,
                        callbacks=[early_stopping, checkpointer])

    # Evaluate model
    print("Evaluate...")
    score, accuracy = model.evaluate(test_dataset)
    print('Test score:', score)
    print('Test accuracy:', accuracy)

    # Predictions and metrics
    y_pred = model.predict(test_dataset)
    y_pred_binary = (y_pred > 0.5).astype('int32')
    test_labels_binary = np.concatenate([y for x, y in test_dataset], axis=0)

    print("Accuracy-sklearn:", accuracy_score(test_labels_binary, y_pred_binary))
    print(classification_report(test_labels_binary, y_pred_binary))
    plot_auc(test_labels_binary, y_pred_binary, title='ROC Curve')

    return history


def plot_auc(y_true, y_pred, title='ROC Curve'):
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
    plt.show()

def plot_performance(history=None, figure_directory='./output', ylim_pad=[0, 0]):
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
        plt.savefig(figure_directory + "/history")

    plt.show()




if __name__ == "__main__":
    # Assuming you have a list of files to iterate over, replace with your actual paths
    file_name = "input/ArSAS.txt"

    print(f'\nProcessing Dataset: {file_name}')

    train_sequences, test_sequences, train_labels, test_labels = prepare_data(
        file_name=file_name,
        max_sent_len=max_sent_len,
        max_features=max_features,
        random_state=random_state
    )
    train_dataset, test_dataset = convert_to_tf_datasets(
        train_sequences, test_sequences,
        train_labels, test_labels,
        batch_size=batch_size
    )

    history = NN_nets_r0(
        train_dataset, test_dataset,
        max_features=max_features,
        max_sent_len=max_sent_len,
        epochs=epochs,
        embeddings_dim=embeddings_dim,
    )

    plot_performance(history=history)

    # NN_nets_r0(file_name, max_sent_len, )





