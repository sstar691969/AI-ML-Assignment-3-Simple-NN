# AI-ML-Assignment-3-Simple-NN
William Anderson/AD331


MNIST DATA SET - numbers machine learning - neuron/ Neural Networks
Frameworks/imports:
import pandas as pd


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPooling2D,Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

A summary of the model architecture (number of layers, neurons, and activation functions used):
Total of neurons for input: 28*28=784 pixels.
(2)Inner layers/ hidden layers pass the numbers 0 thru 1 of gray or white to output. Then process of 
backpropagation is used to adjust the weights between neurons. The weights are update to reduce error and 
improve the next prediction like: A final prediction of classification of the image of a number or example
as a cat or a dog.
Training or Testing: 
During training, the network is shown examples of images numbers and then 
learn to recognize patterns in them.
After training,the network is tested on new data to check its performance.
The better the network is trained, the more accurately it will predict new data.
finally trained and tested images to correctly identify the selected number or cat or dog.

The final test set accuracy achieved by your model.
Then comes activation and the final analysis. My final output for this
assignment had a crash which I tried to resolve due to CPU GPU on mac pro.
