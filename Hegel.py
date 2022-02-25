from distutils.command.build import build
import pandas as pd
import numpy as np
import matplotlib as plt
import os
import time
import tensorflow as tf
import torch
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout


print(os.listdir("checkpoint"))
latest = tf.train.latest_checkpoint("checkpoint")
print(latest)

#read the data:
#process text
text = open("Hegel.txt", "rb").read().decode(encoding="utf-8")
vocab =  sorted(set(text))
#print(f"Length of text: {len(text)}")
#print(f"{len(vocab)}unique characters")


#printing the first 250 characters of thet text:
#print(text[:250])


#vectorising the text:

#creating a  mapping from unique characters to indices
#we are mapping strings to a numerical representation. We must create two looukp tables, one mapping
#characters to numbers, and another for numbers to characters


char2idx =  {u:i for i, u in enumerate(vocab)}
idx2char =  np.array(vocab)

text_as_int =  np.array([char2idx[c] for c in text])

#show how the first 13 characters from the text are mapped to integers
#print(f" {repr(text[1:13])} --- characters mapped to int -----> {repr(text_as_int[:13])}")


#creating training examples and targets:

#maximum length sentence you want for a single imput of characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length + 1)

#creating training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

#for i in char_dataset.take(5):
 #   print(idx2char[i.numpy()])



#For each input sequence, the corresponding targets contain the same length of text, except shifted
# one character to the right. Hence, that’s why we should divide the text into chunks of ‘seq_length +1’.
# If input sequence would be “Hell”,  and the target sequence “ello”. tf.data.Dataset.from_tensor_slices function
# converts the text vector into a stream of character indices.
#As a result, we can see that the word ‘Phenomenology’ is sliced by each character.

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

#for item in sequences.take(5):
 #   print(repr("".join(idx2char[item.numpy()])))


#implementing batches to convert individual characters to sequences of desired sizes:

#for each sequence duplciate it and shift it to form the imput and target text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

#use the map method to apply the function to each batch
dataset = sequences.map(split_input_target)

#print the example of input and example of target

#for input_example, target_example in dataset.take(1):
 #   print("input data:", repr("".join(idx2char[input_example.numpy()])))
  #  print("target data:", repr("".join(idx2char[target_example.numpy()])))



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
    #print(example_batch_predictions.shape, "batch size, seq length, vocab size")


#creating sample indices for the first example batch
#draws samples from the categorical distribution
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples =1)

#Use squeeze to discard the dimensions of length one out of the shape of a stated
sampled_indices = tf.squeeze(sampled_indices, axis=1).numpy()

#print("input:", repr("".join(idx2char[input_example_batch[0]])))
#print("input:", repr("".join(idx2char[sampled_indices])))


#training the model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
#print("prediction shape:", example_batch_predictions.shape,  "batch size, seq length, vocab size")
#print("scaler_loss", example_batch_loss.numpy().mean())



Epochs = 100

checkpoint_filepath = 'checkpoint/checkpoint {epoch:02d}'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    mode='min',
    save_best_only=True)

#stops training the model if after 10 epochs there is no improvement in the loss
early_stop =  tf.keras.callbacks.EarlyStopping(monitor= "loss", patience=10, verbose=1)


#logs epoch, loss into a csv file. The csv file will then be used to plot an epoch versus Loss graph
log_csv = tf.keras.callbacks.CSVLogger("logs.csv", separator=",", append=False)


# Model weights are saved at the end of every epoch, if and only if the model is the best seen
# so far. These models weights will located in the "checkpoint <epoch number>" directories, which are all
# subdirectories of the "checkpoint directory"
#model.compile(optimizer="adam",loss=loss)


#history = model.fit(dataset, epochs=Epochs, callbacks=[checkpoint_callback, early_stop, log_csv])

model.save("Hegel_Augmented_model.h5")
print(model.summary())
# The model weights (that are considered the best) are loaded into the model.


#generating text using the build model

model = build_model(len(vocab), embedding_dim, rnn_units, batch_size=1)
#model.load_weights(tf.train.latest_checkpoint("checkpoint/checkpoint 53"))

#model.build(tf.TensorShape([1, None]))



def Hegel(model, start_string):
    #generating the next step using the learned model
    num_generate = 1000

    #converting start string to nubers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    #empty list to store results
    text_generated=[]

    temp = 1.0

    #batch size ==1
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        #remove batch dim
        predictions = tf.squeeze(predictions, 0)


        #use cat dist to predict char returned by model

        predictions = predictions/temp
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        #pass the predicted character as the next input to the model
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + " ".join(text_generated))



print(Hegel(model,start_string = "Spirit" ))
