# test3.py
# DEEP LEARNING IN PYTHON
# Ch 6: TensorFlow
# Matrix multiplication
#

# nascondi warnings di tensorflow per compilazione
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# importa TensorFlow
import tensorflow as tf
# importa numpy
import numpy as np

# dichiara un placeholder (dati in input)
A = tf.placeholder(
    dtype=tf.float32, # tipo dati
    shape=(5,5),      # dimensioni 5x5 [opzionale]
    name='A'          # nome simbolico 'A' [opzionale]
)

# dichiara un altro placeholder di tipo float32
v = tf.placeholder( tf.float32 )

# crea un nodo u che compie la moltiplicazione matriciale tra A e v
u = tf.matmul( A, v )

# crea una sessione di TensorFlow
with tf.Session() as session:
    # calcola l'output del nodo "u" eseguendo la sessione
    # i valori per i placeholder sono assegnati tramite
    # l'argomento "feed_dict", ricordando le dimensioni corrette
    # per il calcolo matriciale
    _A = np.random.randn(5, 5) # assegna ad A valori random (matrice 5x5)
    _v = np.random.randn(5, 1) # assegna a v valori random (vettore colonna 5x1)
    output = session.run(
        u, # nodo da eseguire
        feed_dict={A: _A, v: _v}
    )
    # stampa il risultato
    print("A: ", _A, "\nv: ", _v, "\nu:", output, "\n", type(output))
