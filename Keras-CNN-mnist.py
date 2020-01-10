#Import Libraries
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical


#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
