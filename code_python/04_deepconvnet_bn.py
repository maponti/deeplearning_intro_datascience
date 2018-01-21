## Codigo exemplo para a Escola Avancada de Matematica, minicurso Deep Learning
##
## Exemplo de rede neural convolucional profunda com Batch normalization
##
## Moacir A. Ponti (ICMC/USP), Janeiro de 2018
##
## Referencia: Everything you wanted to know about Deep Learning for Computer Vision but were afraid to ask. Moacir A. Ponti, Leonardo S. F. Ribeiro, Tiago S. Nazare, Tu Bui and John Collomosse
##
import tensorflow as tf

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# 1) Definir arquitetura
# placeholder sao variaveis que podem receber um numero indeterminado de elementos, ainda nao definidos

# numero de neuronios / filtros por camada
L1 = 32
L2 = 128

# matriz de entrada (None significa indefinido)
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # tamanho original 28x28x1

# matriz de pesos para cada camada

# Camada 1 convolucional seguida de maxpooling
W1 = tf.Variable(tf.truncated_normal([5,5,1, L1], stddev=0.1)) # filtros 5x5x1, L1 neuronios/kernels
b1 = tf.Variable(tf.ones([L1])/10)      # bias de L1 feature maps

# Camada 2 densa
# por isso sera preciso que haja um peso por valor do feature map
# produzido pela camada anterior, entao teremos uma imagem 28x28
# seguida de maxpooling a imagem ficara com 14x14
W2 = tf.Variable(tf.truncated_normal([14*14*L1, L2], stddev=0.1)) # L1 x L2 neuronios
b2 = tf.Variable(tf.ones([L2])/10)      

# Camada 3 densa, de saida com 10 neuronios
W3 = tf.Variable(tf.truncated_normal([L2, 10], stddev=0.1)) # L2 neuronios x 10 classes
b3 = tf.Variable(tf.ones([10])/10)      # 10 classes

# saida
Y = tf.placeholder(tf.float32, [None, 10]) # rotulos (de treinamento!)


# Modelo que ira gerar as predicoes
# Mutiplicacao matricial, soma de vetor, funcao de ativacao Softmax
# funcao_ativacao(WX + b) = softmax(WX+b)

# Define outras variaveis, algumas como placeholders
lRate = tf.placeholder(tf.float32) # learning rate
pKeep = tf.placeholder(tf.float32) # probabilidade manter (dropout)

# flag para o BN (normalizacao por batch)
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)

batchSize = 100

# Funcao que aplica a normalizacao em batch usando medias moveis de media e variancia
# Thanks Marting Gorner
def batchnorm(Vlogits, isTest, iteration, offset, convlayer=False):
	BNepsilon = 0.00001

	# se for convolucional, preciso pegar valores das 3 dimensoes do feature map
	if convlayer:
		mean, var = tf.nn.moments(Vlogits, [0,1,2])
	else:
		mean, var = tf.nn.moments(Vlogits, [0])

	# media movel exponencial, para na iteracao atual
	expMovingAvg = tf.train.ExponentialMovingAverage(0.999, iteration)
	print(str(iteration) + " " + str(expMovingAvg))
	movingAverage = expMovingAvg.apply([mean, var])
	m = tf.cond(isTest, lambda: expMovingAvg.average(mean), lambda:mean)
	v = tf.cond(isTest, lambda: expMovingAvg.average(var), lambda:var)

	VBN = tf.nn.batch_normalization(Vlogits, m, v, offset, None, BNepsilon)

	return VBN, movingAverage

# Funcao para operacoes sem BN
def no_batchnorm(Vlogits, isTest, iteration, offset, convlayer=False):
	return VBN, tf.no_op()
	


# vetor de entrada (agora como imagem)
X1 = tf.reshape(X, [-1, 28, 28, 1])

## representacoes intermediarias
# feature maps da camada convolucional
X2 = tf.nn.conv2d(X1,W1, strides=[1,2,2,1], padding="SAME")
X2, medvar1 = batchnorm(X2, tst, iter, b1, convlayer=True)
X2 = tf.nn.relu(X2) # repare que nao tem bias, adicionado acima

# passagem de feature maps 2d para um vetor 1d
# achatando a matriz (flatten) em um vetor
X2 = tf.contrib.layers.flatten(X2)

# neuronios da camada densa, com dropout
X3 = tf.matmul(X2, W2)
X3, medvar2 = batchnorm(X3, tst, iter, b2, convlayer=False)
X3 = tf.nn.relu(tf.nn.bias_add(X3, b2))
X3 = tf.nn.dropout(X3, pKeep)

# formula para as predicoes
X4 = tf.nn.bias_add(tf.matmul(X3, W3), b3)
Y_ = tf.nn.softmax(X4)

medvar = tf.group(medvar1, medvar2)

# 2) Funcao de custo: informa quao longe estamos da solucao desejada
# Entropia cruzada e' uma das mais usadas, relacoes com divergencia de Kullback-Leibler
# - [ Y * log(Y_) ]
crossEnt = tf.nn.softmax_cross_entropy_with_logits(logits=X4, labels=Y)
crossEnt = tf.reduce_mean(crossEnt)*batchSize

# Outra medida de qualidade pode ser a acuracia
correctPred = tf.equal( tf.argmax(Y_,1), tf.argmax(Y,1) )
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# 3) Metodo de otimizacao e taxa de aprendizado
optMethod = tf.train.AdamOptimizer(lRate)
trainProcess = optMethod.minimize(crossEnt)

# Tudo pronto, agora podemos executar o treinamento
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

iterations = 201
# Dataset
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# 4) Treinamento por iteracoes, cada iteracao realiza um
#    feed-forward na rede, computa custo, e realiza backpropagation
for i in range(iterations):
    # carrega batch de dados com respectivas classes
    batX, batY = mnist.train.next_batch(batchSize)

    # define dicionrio com pares: (exemplo,rotulo)
    # e parametros de treinamento taxa de aprendizado e probabilidade dropout
    trainData = {X: batX, Y: batY, lRate:0.0025, pKeep:0.5, tst:False}
    batchNorm = {X: batX, Y: batY, iter:i, pKeep:0.5, tst:False}

    # executa uma iteracao com o batch carregado
    sess.run(trainProcess, feed_dict=trainData)
    sess.run(medvar, feed_dict=batchNorm)

    # computa acuracia no conjunto de treinamento e funcao de custo
    # (a cada 5 iteracoes)
    if (i%5 == 0):
        acc, loss = sess.run([accuracy, crossEnt], feed_dict=trainData)
        print(str(i) + " Loss ="+str(loss) + " Train.Acc="+str(acc))


# 5) Valida o modelo nos dados de teste (importante!)
testData = {X: mnist.test.images, Y: mnist.test.labels, tst:True, pKeep:1.0}
accTest, lossTest = sess.run([accuracy, crossEnt], feed_dict=testData)

print("\nTest\tAccuracy="+str(accTest))
print("\tLoss="+str(lossTest))

