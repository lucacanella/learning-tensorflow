# test4.py
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
u = tf.Variable(20.0)

# crea la funzione di costo
cost = u*u + u + 1.0

# crea l'ottimizzatore a discesa del gradiente:
#   assegna rateo d'apprendimento 0.3
#   minimizza la funzione 'cost' definita precedentemente
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

# crea un nodo che inizializza le variabili di tf
init = tf.initialize_all_variables()

#crea la sessione
with tf.Session() as session:
    session.run(init) # esegui il nodo di inizializzazione
    # esegui i cicli di training
    for i in range(12):
        # esegui l'op. train_op
        session.run(train_op)
        # stampa i risultati parziali
        print("i = %d, cost %.3f, u = %.3f" % (i, cost.eval(), u.eval()))
