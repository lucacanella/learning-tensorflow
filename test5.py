# test5.py
# XOR
# Implementing xor
#

#imports
import os
import tensorflow as tf
import numpy as np

# nascondi warnings di tensorflow per compilazione
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_shape  = (0, 2)
labels_shape = (0, 1)

# variabile di input
x = tf.placeholder(dtype=tf.float32, name="x-in")
# labels
y = tf.placeholder(dtype=tf.float32, name="y-out")

weights1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="m1")
bias1    = tf.Variable(tf.zeros([2]), name="b1")
weights2 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="m2")
bias2    = tf.Variable(tf.zeros([2]), name="b2")

layer1  = tf.sigmoid(tf.matmul(x, weights1) + bias1)
outputl = tf.sigmoid(tf.matmul(layer1, weights2) + bias2)

cost = tf.reduce_mean((y - outputl) ** 2)

decay_global_step = tf.Variable(0, trainable=False)
decaying_learning_rate = tf.train.exponential_decay(learning_rate=0.55, global_step=decay_global_step, decay_steps=3500, decay_rate=0.95, staircase=True)

# crea l'ottimizzatore a discesa del gradiente:
#   assegna rateo d'apprendimento 0.1
#   minimizza la funzione 'cost' definita precedentemente
train_op = tf.train.GradientDescentOptimizer(decaying_learning_rate).minimize(cost, global_step=decay_global_step)

# crea un nodo che inizializza le variabili di tf
init = tf.global_variables_initializer()

X = [
    [ 0,0 ],
    [ 0,1 ],
    [ 1,0 ],
    [ 1,1 ]
]
Y = [
    [0],
    [1],
    [1],
    [0]
]

#crea la sessione
with tf.Session() as session:
    session.run(init) # esegui il nodo di inizializzazione
    # esegui i cicli di training
    for i in range(100000):
        # esegui l'op. train_op
        session.run(train_op, feed_dict={ x: X, y: Y })
        # stampa i risultati parziali
        if i % 1000 == 0:
            tCost = session.run(cost, feed_dict={ x: X, y: Y })
            lr = decaying_learning_rate.eval()
            print("i = %6d, cost: %.5f [current lr: %.5f] " % (i, tCost, lr))
