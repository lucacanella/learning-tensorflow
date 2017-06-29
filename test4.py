# test3.py
# DEEP LEARNING IN PYTHON
# Ch 6: TensorFlow
# Simple optimization problem
#

#imports
import os
import tensorflow as tf
import numpy as np

# nascondi warnings di tensorflow per compilazione
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# crea una variabile
u = tf.Variable()