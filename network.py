from turtle import hideturtle, shape
import numpy
import math
import sklearn
from sklearn.utils import shuffle
import pandas
import csv


# If response close to 0 then H0, if response close to 1 then H1
class Network :
    __layers = []
    __neurons = [[]]
    __biases = [[]]
    __weights = [[[]]]

    fitness: float = 0

    def __init__(self, layers):

        self.__layers = [layers[i] for i in range(len(layers))]
        self.__initNeurons()
        self.__initBiases()
        self.__initWeights()
    
    def __initNeurons(self) :
        self.__neurons = [ [float(0) for i in range(int(self.__layers[i]))] for i in range(len(self.__layers)) ]
        # print('\n')
        # print(self.__neurons)
    
    def __initBiases(self):
        self.__biases = [ [float(0) for _ in range(self.__layers[i])] for i in range(1, len(self.__layers)) ]
        # print('\n')
        # print(self.__biases)
    
    def __initWeights(self):
        self.__weights = [[[float(self.generateRandomNumber(self.__layers[i-1], self.__layers[i])) for _ in range(self.__layers[i-1])] for _ in range(self.__layers[i])] for i in range(1, len(self.__layers))]
        # print('\n')
        # print(self.__weights)

    def sigmoid_prime(self, z):
        return self.__activate(z) * (1 - self.__activate(z))
        
    def __activate(self, value_to_reduce):
        return float(1 / (1 + math.exp(-value_to_reduce)))
        
    
    def generateRandomNumber(self, n, m):
        number = numpy.random.normal(loc = 0, scale = (1 / (n + m)), size=1)
        number = number[0]
        return number
    
    def feedForward(self, inputs):
        
        for i in range(len(inputs)):
            self.__neurons[0][i] = inputs[i]
        
        for i in range(1, len(self.__layers)):
            prev_layer = i-1
            for j in range(len(self.__neurons[i])):
                neuron_value = float(0)
                for k in range (len(self.__neurons[prev_layer])):
                    neuron_value += self.__weights[prev_layer][j][k] * self.__neurons[prev_layer][k]
                self.__neurons[i][j] = self.__activate(neuron_value + self.__biases[prev_layer][j])

        return self.__neurons[1], self.__neurons[-1]


    def cross_entropy(self, y, y_hat, label=None):
        def safe_log(x): return 0 if x == 0 else numpy.log(x)

        total = 0
        # if (label == 0 ):
        #     total = -safe_log(y_hat[0])
        #     return total
        
        # else:
        #     total = -safe_log(y_hat[1])
        #     return total
        for curr_y, curr_y_hat in zip(y, y_hat):
            total += (curr_y * safe_log(curr_y_hat) + (1 - curr_y) * safe_log(1 - curr_y_hat))
            
        
        return - total / len(y)





    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None, label=None):
        
        samples = len(training_data)
        if test_data:
            # test_data = list(test_data)
            n_test = len(test_data)
        for j in range(epochs):
            # Stochastic
            shuffle(training_data)
            
            # Splitting data into batches
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, samples, mini_batch_size)]

            # For each batch created
            for mini_batch in mini_batches:
                # print(mini_batch)
                self.__update_mini_batch(mini_batch, learning_rate, label)

            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"-------------------------------------------Epoch {j} complete-------------------------------------------")
    

    def __update_mini_batch(self, mini_batch, learning_rate, label=None):
        temp_biases = [numpy.zeros(numpy.shape(biases)) for biases in self.__biases]
        # print('TEMP BIASES\n', temp_biases, '\n')
        temp_weights = [numpy.zeros(numpy.shape(weights)) for weights in self.__weights]
        # print('TEMP WEIGHTS\n', temp_weights, '\n')
        
        
        for x1, x2 in mini_batch.to_numpy():
            # print(mini_batch)
            # Calling backpropagation
            self.backprop(x1, x2, len(mini_batch), learning_rate, label)

    

    def backprop(self, x, y, batch_size, learning_rate, label=None):
        W_update = [numpy.zeros(numpy.shape(b)) for b in self.__biases]
        B_update = [numpy.zeros(numpy.shape(w)) for w in self.__weights]
        inputs = [x, y]
        # print(inputs)

        # print(inputs)
        hidden_layer, outputs = self.feedForward(inputs)
        # print('\nHidden Layer ->\n{}\nHidden Layer Shape -> {}\n'.format(hidden_layer, numpy.shape(hidden_layer)))
        # print(inputs,'-->', outputs)

        # outputs[0] = z 
        # 1 - outputs[0] = 1-z
        
        if (label == 0):
            p = [1, 0.00000001]
            q = [1 - outputs[0], outputs[0]]
            
        else: 
            p = [0.0000001, 1]
            q = [1 - outputs[0], outputs[0]]
        # q = [0.25, 0.75]

        E1 = self.cross_entropy(p, q)
        # E1 = abs(outputs[0] - 1)
        # Watch
        # BIASES ALWAYS 0 --> FIX
        # https://medium.com/dataseries/back-propagation-algorithm-and-bias-neural-networks-fcf68b5153
        # https://www.askpython.com/python/examples/backpropagation-in-python#:~:text=%20Implementing%20Backpropagation%20in%20Python%20%201%20Import,Split%20Dataset%20in%20Training%20and%20Testing%20More%20
        
        dW1 = E1 * outputs[0] * (1 - outputs[0]) # THIS IS FOR THE OUTPUT LAYER ONLY 
        dB1 = outputs[0] - label 
        # print('\nWeights[-1]\n', self.__weights[-1], '\n')
        E2 = numpy.dot(dW1, numpy.transpose(self.__weights[-1]))
        # print('\n E2 ->\n{}\nE2.Shape -> {} \n'.format(E2, numpy.shape(E2)))
        
        
        hidden_layer = [hidden_layer]
        # print('\nHidden Layer ->\n{}\nHidden Layer Shape -> {}\n'.format(hidden_layer, numpy.shape(hidden_layer)))
        # print('\n1 - Hidden Layer ->\n{}\n1 - Hidden Layer Shape -> {}\n'.format(numpy.ones(shape=numpy.shape(hidden_layer)) - hidden_layer, numpy.shape(numpy.ones(shape=numpy.shape(hidden_layer)) - hidden_layer)))
        
        dW2 = E2 * numpy.transpose(hidden_layer * (numpy.ones(shape=numpy.shape(hidden_layer)) - hidden_layer))

        inputs = [inputs]
        # print('\n', inputs)
        W2_update = numpy.dot(numpy.transpose(hidden_layer), dW1) / batch_size
        W1_update = numpy.dot(dW2, inputs) / batch_size
        W2_update = [weight for weight in W2_update[:,0]]


        # print('\nW1_update ->\n{}\nW1_update Shape -> {}'.format(W1_update, numpy.shape(W1_update)))
        # print('\nW2_update ->\n{}\nW2_update Shape -> {}'.format(W2_update, numpy.shape(W2_update)))
        # W1_update = np.dot(X_train.T, dW2) / N
        # WE NEED DIFFERENT dW1 FOR INTERMEDIATE LAYERS
        W_update[0] = W1_update.tolist()
        W_update[1] = [W2_update]
        

        print('Output -> {}\t\tE1 -> {}'.format(numpy.round(outputs[0], 4), numpy.round(E1, 4)))


        
        for i in range (len(self.__layers) - 1):
            self.__biases[i] = self.__biases[i] - numpy.full(shape=numpy.shape(self.__biases[i]), fill_value=(learning_rate * dB1))
        
        # print('\nBIASES', B_update[0])
        # print('\nBIASES', B_update[1])
            
        # print('\nW_UPDATE\n W_update[0] --->\n{}\nW_update[0] Shape ---> {}\nW_update[1] --->\n{}W_update[1] Shape ---> {}\n'.format( W_update[0],numpy.shape(W_update[0]), W_update[1], numpy.shape(W_update[1])))

        for i in range (len(self.__layers) - 1):
            # TODO code for output node is different than intermediate node
            self.__weights[i] = self.__weights[i] - (W_update[i] * numpy.full(shape=numpy.shape(W_update[i]), fill_value=learning_rate))
        
        # print('\n FINAL WEIGHTS\n', self.__weights)
        # print('\n FINAL BIASES\n', self.__biases)

        # print('\nWEIGHTS', W_update[0])
        # print('\nWEIGHTS', W_update[1])

        # print('\n',W_update, '\n')

        # print('\n-------------------------------------------------------------------------------------------------------------------------------------\n')

        



                          
if __name__ == '__main__':
    network = Network([2,20,1])
    
    # # print(len(train_data))
    # # print('\n',network.cross_entropy([1, 0.00001], [0.86, 0.14]))
    # train_data = pandas.read_csv('./train_samples_F0.csv')

    # network.SGD(training_data=train_data, epochs=100, mini_batch_size=10, learning_rate=0.01)

    train_data = pandas.read_csv('./train_samples_F1.csv')
    network.SGD(training_data=train_data, epochs=100, mini_batch_size=10, learning_rate=0.01, label=1)

    hidden_layer, outputs = network.feedForward([-0.23166163020985503,-1.1604970924720286])
    print(outputs[0])
    



