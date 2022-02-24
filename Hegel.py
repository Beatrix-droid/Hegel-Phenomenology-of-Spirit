from distutils.command.build import build
import pandas as pd
import numpy as np
import matplotlib as plt
import os
import time
import tensorflow as tf
import torch
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#read the data:
#process text
text = open("Hegel.txt", "rb").read().decode(encoding="utf-8")
vocab =  sorted(set(text))
print(f"Length of text: {len(text)}")
print(f"{len(vocab)}unique characters")


#printing the first 250 characters of thet text:
print(text[:250])


#vectorising the text:

#creating a  mapping from unique characters to indices
#we are mapping strings to a numerical representation. We must create two looukp tables, one mapping
#characters to numbers, and another for numbers to characters


char2idx =  {u:i for i, u in enumerate(vocab)}
idx2char =  np.array(vocab)

text_as_int =  np.array([char2idx[c] for c in text])

#show how the first 13 characters from the text are mapped to integers
print(f" {repr(text[1:13])} --- characters mapped to int -----> {repr(text_as_int[:13])}")


#creating training examples and targets:

#maximum length sentence you want for a single imput of characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length + 1)

#creating training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])



#For each input sequence, the corresponding targets contain the same length of text, except shifted
# one character to the right. Hence, that’s why we should divide the text into chunks of ‘seq_length +1’.
# If input sequence would be “Hell”,  and the target sequence “ello”. tf.data.Dataset.from_tensor_slices function
# converts the text vector into a stream of character indices.
#As a result, we can see that the word ‘Phenomenology’ is sliced by each character.

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

for item in sequences.take(5):
    print(repr("".join(idx2char[item.numpy()])))


#implementing batches to convert individual characters to sequences of desired sizes:

#for each sequence duplciate it and shift it to form the imput and target text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

#use the map method to apply the function to each batch
dataset = sequences.map(split_input_target)

#print the example of input and example of target

for input_example, target_example in dataset.take(1):
    print("input data:", repr("".join(idx2char[input_example.numpy()])))
    print("target data:", repr("".join(idx2char[target_example.numpy()])))



#creating training batches:
Batch_size = 100

Buffer_size = 10000

dataset = dataset.shuffle(Buffer_size).batch(Batch_size, drop_remainder=True)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


#building the model
#must shuffle the data and pack it into batches first:

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[Batch_size, None]),
                                 tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
                                 tf.keras.layers.Dense(vocab_size)
                                 ])
    return model

# for each character the model looks at the embedding, runs the GRU one timestep with the embedding as imput,
#applies the dense layer to generate logits predicting the log-likehood of the next character

model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size = Batch_size)

#trying the model:

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions =  model(input_example_batch)
    print(example_batch_predictions.shape, "batch size, seq length, vocab size")


#creating sample indices for the first example batch
#draws samples from the categorical distribution
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples =1)

#Use squeeze to discard the dimensions of length one out of the shape of a stated
sampled_indices = tf.squeeze(sampled_indices, axis=1).numpy()

print("input:", repr("".join(idx2char[input_example_batch[0]])))
print("input:", repr("".join(idx2char[sampled_indices])))


#training the model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("prediction shape:", example_batch_predictions.shape,  "batch size, seq length, vocab size")
print("scaler_loss", example_batch_loss.numpy().mean())



#execute training to be run tonight:

Epochs = 100

checkpoint_filepath = '/tmp/checkpoint'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
model.compile(optimizer="adam",loss=loss)


history = model.fit(dataset, epochs=Epochs, callbacks=[checkpoint_callback])


# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)
