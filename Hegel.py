import pandas as pd
import numpy as np
import matplotlib as plt
import os
import time
import tensorflow as tf
import torch
import keras
#from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#read the data:
#process text
text = open("Hegel.txt", "rb").read().decode(encoding="utf-8")
vocab =  sorted(set(text))
print(f"Lenght of text: {len(text)}")
print(f"{len(vocab)}unique characters")
