## Codigo exemplo para a Escola Avancada de Matematica, minicurso Deep Learning
##
## Exemplo de rede neural com uma unica camada
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

# matriz de entrada (None significa indefinido)
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # tamanho original 28x28x1

# matriz de pesos
W = tf.Variable(tf.zeros([784, 10])) # 784 features x 10 classes
b = tf.Variable(tf.zeros([10]))      # bias de 10 classes

# saida
Y = tf.placeholder(tf.float32, [None, 10]) # rotulos (de treinamento!)

init = tf.global_variables_initializer()

# Modelo que ira gerar as predicoes
# Mutiplicacao matricial, soma de vetor, funcao de ativacao Softmax
# funcao_ativacao(WX + b) = softmax(WX+b)

# formula para as predicoes
Y_ = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

# 2) Funcao de custo: informa quao longe estamos da solucao desejada
# Entropia cruzada e' uma das mais usadas, relacoes com divergencia de Kullback-Leibler
# - [ Y * log(Y_) ]
crossEnt = -tf.reduce_sum(Y * tf.log(Y_))

# Outra medida de qualidade pode ser a acuracia
correctPred = tf.equal( tf.argmax(Y_,1), tf.argmax(Y,1) )
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# 3) Metodo de otimizacao e taxa de aprendizado
lrate = 0.003
optMethod = tf.train.GradientDescentOptimizer(lrate)
trainProcess = optMethod.minimize(crossEnt)

# Tudo pronto, agora podemos executar o treinamento
sess = tf.Session()
sess.run(init)

# Define outras variaveis
batchSize = 64
iterations = 1001

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
    # (a cada 10 iteracoes)
    if (i%10 == 0):
        acc, loss = sess.run([accuracy, crossEnt], feed_dict=trainData)
        print(str(i) + " Loss ="+str(loss) + " Train.Acc="+str(acc))


# 5) Valida o modelo nos dados de teste (importante!)
testData = {X: mnist.test.images, Y: mnist.test.labels}
accTest = sess.run(accuracy, feed_dict=testData)

print("\nAccuracy Test="+str(accTest))




