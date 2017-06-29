# test1.py
# Prove di base con tensorflow e python
#

# nascondi warnings di tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# importa tensorflow
import tensorflow as tf

#nodo costante
node1 = tf.constant(3.0, dtype = tf.float32)
#nodo costante
node2 = tf.constant(4.0) # also tf.float32 implicitly
#nodo placeholder (variabile da assegnare al momento della run)
ph = tf.placeholder(tf.float32)

#nodo operazione
addernode = node1 + node2 * ph

#crea una sessione ed esegui il nodo, specificando un valore per il placeholder
sess = tf.Session()
resultNode = sess.run(addernode, { ph: 2.0 })

#stampa
print(resultNode)

#modello lineare

sess2 = tf.Session()

W = tf.Variable( [.3],  dtype = tf.float32 )
b = tf.Variable( [-.3], dtype = tf.float32 )
x = tf.placeholder( tf.float32 )
linear_model = W * x + b

#inizializza le variabili
init = tf.global_variables_initializer()
sess2.run(init)
#esegui per x = 1, 2, 3 e 4
print(sess2.run(linear_model, { x: [1, 2 ,3 ,4] }))


