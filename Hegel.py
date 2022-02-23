from distutils.command.build import build
import pandas as pd
import numpy as np
import matplotlib as plt
import os
import time
import tensorflow as tf
import torch
import keras
from tensorflow import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#read the data:
#process text
text = open("Hegel.txt", "rb").read().decode(encoding="utf-8")
vocab =  sorted(set(text))
print(f"Lenght of text: {len(text)}")
print(f"{len(vocab)}unique characters")


#printing the frist 250 characters of thet text:
print(text[:250])


#vectorising the text:

#creatinga  mapping from unique characters to indices
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


#implementing batches to convert indivdual characters to sequences of desired sizes:

#for each sequence duplciate it and shift it to form the imput and target text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

#use teh map method to apply the function to each batch
dataset = sequences.map(split_input_target)

#print the example of input and example of target

for input_example, target_example in dataset.take(1):
    print("input data:", repr("".join(idx2char[input_example.numpy()])))
    print("target data:", repr("".join(idx2char[target_example.numpy()])))



#creating training batches:
Batch_size = 100

Buffer_size = 10000

dataset = dataset.shuffle(Buffer_size).batch(Batch_size, drop_remainder=True)


#building the model
#must shuffle the data and pack it into batches first:

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[Batch_size, None]),
                                 tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
                                 tf.keras.layers.Dense(vocab_size)
                                 ])
    return model

# for each character the model looks at the embedding, runs the GRU one timestep witht eh embedding as imput,
#applies the dense layer to generate logits predicting the log-likehood of the next character

model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size = Batch_size)
