from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import preprocess
from keras.models import load_model
from train import train
import numpy as np
import pandas as pd
import sys

# CUSTOM DATA PREPROCESSING
data = pd.read_csv("./data/3authors.csv")

# remove 4 million rows
# data.drop(index=data.index[:300000], axis=0, inplace=True)

# hyper-paramaters
MAX_LENGTH = 100
NUM_WORDS = 10000

cli_args = sys.argv[1:]
if cli_args[0] == "train":
    train_padded, train_labels, test_padded, test_labels, _ = preprocess(
        data, MAX_LENGTH=MAX_LENGTH, NUM_WORDS=NUM_WORDS)

    train(
        train_padded=train_padded,
        train_labels=train_labels,
        test_padded=test_padded,
        test_labels=test_labels,
        num_words=NUM_WORDS,
        max_length=MAX_LENGTH
    )

elif cli_args[0] == "predict":
    _, _, _, _, tokenizer = preprocess(
        data, MAX_LENGTH=MAX_LENGTH, NUM_WORDS=NUM_WORDS)

    # load model
    model = load_model(f"./models/{cli_args[1]}.h5")

    while True:
        text = input("Enter the text to attribute to an author: ")

        # Tokenize the text
        sequences = tokenizer.texts_to_sequences([text])

        # Pad the text
        padded = pad_sequences(sequences, maxlen=MAX_LENGTH)

        # Make the prediction
        prediction = model.predict(padded)

        # Get the class with the highest probability
        predicted_class = np.argmax(prediction)

        print(predicted_class)
