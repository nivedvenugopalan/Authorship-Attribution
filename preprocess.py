from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def split_data(data: pd.DataFrame, train_size: int = 0.8):
    """Splits the data into a training and testing set."""
    # shuffle the data
    data = data.sample(frac=1)

    T_rows = data.shape[0]
    train_size = int(T_rows*train_size)

    train, test = data[0:train_size], data[train_size:]

    return pd.DataFrame(train), pd.DataFrame(test)


def preprocess(data: pd.DataFrame, MAX_LENGTH: int, NUM_WORDS: int):
    train, test = split_data(
        data=data,
        train_size=0.8
    )

    # remove empty rows
    train, test = train.dropna(
        axis=0, how='any'), test.dropna(axis=0, how='any')

    # get train and test messages
    train_messages, train_authors = train['text'], np.array(train['author'])
    test_messages, test_authors = test['text'], np.array(test['author'])

    # Tokenize the messages
    num_words = NUM_WORDS
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_messages)

    # Sequence the messages
    max_length = MAX_LENGTH
    train_sequences = tokenizer.texts_to_sequences(train_messages)
    train_padded = pad_sequences(train_sequences, maxlen=max_length)

    test_sequences = tokenizer.texts_to_sequences(test_messages)
    test_padded = pad_sequences(test_sequences, maxlen=max_length)

    # one-hot encoded vectors
    encoder = OneHotEncoder()
    # fit transform on all author's to avoid any unknown authors in test_labels
    train_labels = encoder.fit_transform(
        np.array(data['author']).reshape(-1, 1))
    test_labels = encoder.transform(np.array(test_authors).reshape(-1, 1))

    # convert them into numpy arrays
    train_labels, test_labels = train_labels.toarray(), test_labels.toarray()

    return train_padded, train_labels, test_padded, test_labels, tokenizer
