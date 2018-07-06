## Codigo exemplo para a Escola de Matematica Aplicada, minicurso Deep Learning
##
## Exemplo de rede neural com camadas ocultas
##
## Moacir A. Ponti (ICMC/USP), Janeiro de 2018
## Referencia: Everything you wanted to know about Deep Learning for Computer Vision but were afraid to ask. Moacir A. Ponti, Leonardo S. F. Ribeiro, Tiago S. Nazare, Tu Bui and John Collomosse
##
import tensorflow as tf

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# 1) Definir arquitetura
# placeholder sao variaveis que podem receber um numero indeterminado de elementos, ainda nao definidos

FM1 = 256
FM2 = 128

# matriz de entrada (None significa indefinido)
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # tamanho original 28x28x1

# matriz de pesos para cada camada

W1 = tf.Variable(tf.truncated_normal([784, FM1], stddev=0.1)) # 784 features x FM1 neuronios
b1 = tf.Variable(tf.ones([FM1])/10)      # bias de FM1 feature maps

W2 = tf.Variable(tf.truncated_normal([FM1, FM2], stddev=0.1)) # FM1 x FM2 neuronios
b2 = tf.Variable(tf.ones([FM2])/10)      

W3 = tf.Variable(tf.truncated_normal([FM2, 10], stddev=0.1)) # FM2 neuronios x 10 classes
b3 = tf.Variable(tf.ones([10])/10)      # 10 classes

# saida
Y = tf.placeholder(tf.float32, [None, 10]) # rotulos (de treinamento!)


# Modelo que ira gerar as predicoes
# Mutiplicacao matricial, soma de vetor, funcao de ativacao Softmax
# funcao_ativacao(WX + b) = softmax(WX+b)

# vetor de entrada
X1 = tf.reshape(X, [-1, 784])

# representacoes intermediarias
X2 = tf.nn.relu(tf.matmul(X1, W1) + b1)
X3 = tf.nn.relu(tf.matmul(X2, W2) + b2)

# formula para as predicoes
X4 = tf.matmul(X3, W3) + b3
Y_ = tf.nn.softmax(X4)

# Define outras variaveis
batchSize = 100

# 2) Funcao de custo: informa quao longe estamos da solucao desejada
# Entropia cruzada e' uma das mais usadas, relacoes com divergencia de Kullback-Leibler
# - [ Y * log(Y_) ]
crossEnt = tf.nn.softmax_cross_entropy_with_logits(logits=X4, labels=Y)
crossEnt = tf.reduce_mean(crossEnt)*batchSize

# Outra medida de qualidade pode ser a acuracia
correctPred = tf.equal( tf.argmax(Y_,1), tf.argmax(Y,1) )
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# 3) Metodo de otimizacao e taxa de aprendizado
lrate = 0.0025
optMethod = tf.train.GradientDescentOptimizer(lrate)
trainProcess = optMethod.minimize(crossEnt)

# Tudo pronto, agora podemos executar o treinamento
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

iterations = 501
# Dataset
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# 4) Treinamento por iteracoes, cada iteracao realiza um
#    feed-forward na rede, computa custo, e realiza backpropagation
for i in range(iterations):
    # carrega batch de dados com respectivas classes
    batX, batY = mnist.train.next_batch(batchSize)

    # define dicionrio com pares: (exemplo,rotulo)
    trainData = {X: batX, Y: batY}

    # executa uma iteracao com o batch carregado
    sess.run(trainProcess, feed_dict=trainData)

    # computa acuracia no conjunto de treinamento e funcao de custo
    # (a cada 5 iteracoes)
    if (i%5 == 0):
        acc, loss = sess.run([accuracy, crossEnt], feed_dict=trainData)
        print(str(i) + " Loss ="+str(loss) + " Train.Acc="+str(acc))


# 5) Valida o modelo nos dados de teste (importante!)
testData = {X: mnist.test.images, Y: mnist.test.labels}
accTest, lossTest = sess.run([accuracy, crossEnt], feed_dict=testData)

print("\nTest\tAccuracy="+str(accTest))
print("\n\tLoss="+str(lossTest))

