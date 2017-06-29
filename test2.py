# test1.
# Modello lineare
# https://www.tensorflow.org/get_started/get_started
#

# nascondi warnings di tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# importa tensorflow e numpy
import numpy as np
import tensorflow as tf

# Parametri del modello
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Input e output del modello
x = tf.placeholder(tf.float32) #input
y = tf.placeholder(tf.float32) #output

#modello lineare
linear_model = W * x + b

# funzione d'errore
loss = tf.reduce_sum(tf.square(linear_model - y)) # somma dei quadrati (differenze tra risultato del modello lineare e l'output atteso)

# Ottimizzatore gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01)

#nodo di training
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# nodo di inizializzazione variabili
init = tf.global_variables_initializer()
# crea la sessione
sess = tf.Session()

# loop di training:
# 1. inizializza variabili
sess.run(init) 
#ciclo di training
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})
  if i%100 == 0: #ogni 100 cicli di training stampa l'errore
    curr_loss = sess.run(loss, {x:x_train, y:y_train}) #calcola l'errore eseguendo il nodo di curr_loss
    print(i, curr_loss) #stampa l'iterazione e l'errore corrente

# valutazione dell'accuratezza
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
#stampa risultati
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))