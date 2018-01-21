## Codigo exemplo para a Escola Avancada de Matematica, minicurso Deep Learning
##
## Exemplo de autoencoder
##
## Moacir A. Ponti (ICMC/USP), Janeiro de 2018
##
## Referencia: Everything you wanted to know about Deep Learning for Computer Vision but were afraid to ask. Moacir A. Ponti, Leonardo S. F. Ribeiro, Tiago S. Nazare, Tu Bui and John Collomosse
##

import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

# 1) Definir arquitetura
Linp = 784
L1 = 128

# matriz de entrada (None significa indefinido)
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # tamanho original 28x28x1

# Encoder (1 camada)
# matriz de pesos para cada camada
# random_normal para dados, truncated_normal para imagens
We = tf.Variable(tf.truncated_normal([Linp, L1], stddev=0.1)) # Linp features x L1 neuronios
be = tf.Variable(tf.truncated_normal([L1])) # bias de L1 feature maps

# Decoder (1 camada)
Wd = tf.Variable(tf.truncated_normal([L1, Linp], stddev=0.1)) # L1 x Linp neuronios
bd = tf.Variable(tf.truncated_normal([Linp]))     

# Modelo que ira gerar as predicoes
# Mutiplicacao matricial, soma de vetor, funcao de ativacao 
# TanH para dados entre -1 e 1, Sigmoidal para 0 a 1
# vetor de entrada
X1 = tf.reshape(X, [-1, Linp])

# representacao latente (code)
# obs:
C1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(X1, We), be))

# formula para as predicoes
X_ = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(C1, Wd), bd))

# Define outras variaveis
batchSize = 100

# 2) Funcao de custo: informa quao longe estamos da solucao desejada
# Nao temos rotulos, entao devemos comparar com a entrada
# | X - X_ |^2 - erro medio quadratico, MSE
batch_mse   = tf.reduce_mean(tf.pow(X1 - X_, 2), 1)
mse   = tf.reduce_mean(tf.pow(X1 - X_, 2))
error = X1 - X_

# 3) Metodo de otimizacao e taxa de aprendizado
lrate = 0.0025
trainProcess = tf.train.RMSPropOptimizer(lrate).minimize(mse)

# Tudo pronto, agora podemos executar o treinamento
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

iterations = 1001
# Dataset
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)





# 4) Treinamento por iteracoes, cada iteracao realiza um
#    feed-forward na rede, computa custo, e realiza backpropagation
for i in range(iterations):
    # carrega batch de dados com respectivas classes
    batX, batY = mnist.train.next_batch(batchSize)
    # define dicionrio com pares: (exemplo,rotulo)
    trainData = {X: batX}
    # executa uma iteracao com o batch carregado
    sess.run(trainProcess, feed_dict=trainData)

    # computa acuracia no conjunto de treinamento e funcao de custo
    # (a cada 5 iteracoes)
    if (i%5 == 0):
        loss = sess.run(mse, feed_dict=trainData)
        print(str(i) + " Loss ="+str(loss))


# 5) Valida o modelo nos dados de teste (importante!)
testData = {X: mnist.test.images}
lossTest = sess.run(mse, feed_dict=testData)

testOriginal= mnist.test.images
testDecoded = sess.run(X_,feed_dict=testData)

print("\nTest Loss="+str(lossTest))
#print(testDecoded[1, ...])
#print(testOriginal[1, ...])

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
	# imagem original
	ax = plt.subplot(2, n, i+1)
	#orig = np.reshape(testOriginal[i], (28,28))
	plt.imshow(testOriginal[i].reshape(28,28))
	#print(orig)
	#plt.imshow(orig)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2, n, i+1+n)
	#recon = np.reshape(testDecoded[i], (28,28))
	#print(recon)
	plt.imshow(testDecoded[i, ...].reshape(28,28))
	#plt.imshow(recon)
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.savefig('mnist_ae1.png')
plt.close()
