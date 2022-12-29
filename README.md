# Authorship Attribution with a Neural Network

This project uses a neural network to classify text messages and determine their authors. The model is trained on a dataset of text messages and their corresponding authors, and is able to predict the author of a given text message.

The sample data for this project is obtained from [Kaggle](https://www.kaggle.com/competitions/spooky-author-identification).

## Dependencies

This project requires the following libraries:

- NumPy
- Pandas
- TensorFlow
- scikit-learn
- Keras

You can install these libraries using `pip` by running the following command:
```pip install numpy pandas tensorflow scikit-learn keras```

## Preprocessing

Before training the model, the text messages are preprocessed to convert them into numerical representations that can be fed into the model. This includes tokenizing the messages and padding them to a consistent length.

## Model

The model used in this project is a long short-term memory (LSTM) network with multiple layers. The model is trained using the Adam optimizer and the categorical cross-entropy loss function.

## Evaluation

The model is evaluated using a test set of text messages and their corresponding authors. The model's performance is measured using the accuracy metric.

## Usage

To use the model, you can run the `main.py` script with the following command line arguments:

- `train`: trains a new model and saves it to a file
- `predict modelname`: loads the model from the specified file and classifies a text message

To classify a text message, you can use the `predict()` method of the trained model. The method takes a single argument, which is the text message to classify. It returns the predicted class, which is the author of the text message.

I hope this readme has been helpful! Let me know if you have any questions or if you need further assistance.
