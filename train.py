from keras.models import save_model
import tensorflow as tf
import numpy as np


def train(train_padded: np.array, train_labels: np.array, test_padded: np.array, test_labels: np.array, num_words: int, max_length: int, EPOCHS: int = 10,  save_path: str = "./models/", model_id: int = 0, ):

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_words, 32, input_length=max_length),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(train_labels.shape[1], activation='softmax')
    ])

    print("Compiling Model")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    # train the model
    _ = model.fit(
        train_padded,
        train_labels,
        epochs=EPOCHS,
        batch_size=32,
        validation_split=0.2
    )

    save_model(model, save_path+f"model{model_id}.h5")

    # evaluate model performance
    loss, accuracy = model.evaluate(test_padded, test_labels)
    print(loss, accuracy)
