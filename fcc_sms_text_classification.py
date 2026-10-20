# import libraries
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install -q tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# get data files
!wget -nc -q https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget -nc -q https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load the data (tab separated: label, message)
train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# ham -> 0, spam -> 1
train_labels = (train_df['label'] == 'spam').astype('float32').values
test_labels = (test_df['label'] == 'spam').astype('float32').values

print(train_df.shape, test_df.shape)
print(train_df['label'].value_counts())
train_df.head()

# Build and train the neural network
tf.keras.utils.set_random_seed(42)

VOCAB_SIZE = 10000
SEQ_LEN = 100

vectorizer = keras.layers.TextVectorization(max_tokens=VOCAB_SIZE,
                                            output_mode='int',
                                            output_sequence_length=SEQ_LEN)
vectorizer.adapt(tf.constant(train_df['message'].tolist()))

model = keras.Sequential([
    keras.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    keras.layers.Embedding(VOCAB_SIZE, 32),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(tf.constant(train_df['message'].tolist()), train_labels,
                    epochs=10,
                    batch_size=64,
                    validation_data=(tf.constant(test_df['message'].tolist()), test_labels),
                    verbose=2)

loss, acc = model.evaluate(tf.constant(test_df['message'].tolist()), test_labels, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
  p = float(model.predict(tf.constant([pred_text]), verbose=0)[0][0])
  prediction = [p, 'spam' if p >= 0.5 else 'ham']
  return (prediction)

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()

