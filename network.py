import numpy
import math
from sklearn.utils import shuffle
import pandas


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


    def cross_entropy(self, y, y_hat):
        def safe_log(x): return 0 if x == 0 else numpy.log(x)

        total = 0
        for curr_y, curr_y_hat in zip(y, y_hat):
            total += (curr_y * safe_log(curr_y_hat) + (1 - curr_y) * safe_log(1 - curr_y_hat))
        return - total / len(y)





    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        
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
                self.__update_mini_batch(mini_batch, eta)

            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"-------------------------------------------Epoch {j} complete-------------------------------------------")
    

    def __update_mini_batch(self, mini_batch, eta):
        temp_biases = [numpy.zeros(numpy.shape(biases)) for biases in self.__biases]
        # print('TEMP BIASES\n', temp_biases, '\n')
        temp_weights = [numpy.zeros(numpy.shape(weights)) for weights in self.__weights]
        # print('TEMP WEIGHTS\n', temp_weights, '\n')
        
        
        for x1, x2 in mini_batch.to_numpy():
            
            # print(mini_batch)
            # Calling backpropagation
            delta_temp_biases, delta_temp_weights = self.backprop(x1, x2, len(mini_batch))
            # print(delta_temp_weights)
            # print('DELTA TEMP BIAS\n', delta_temp_biases, '\n')
            # print(numpy.shape(delta_temp_biases))

            temp_biases = [temp_b + delta_temp_b for temp_b, delta_temp_b in zip(temp_biases, delta_temp_biases)]
            temp_weights = [temp_w + delta_temp_w for temp_w, delta_temp_w in zip(temp_weights, delta_temp_weights)]
        

        self.__weights = [w - (eta) * nw for w, nw in zip(self.__weights, temp_weights)]
        self.__biases = [b - (eta) * nb for b, nb in zip(self.__biases, temp_biases)]

        # print('\n WELCOME', self.__weights)
        # print('\n', self.__biases)
    

    def backprop(self, x, y, batch_size):
        W_update = [numpy.zeros(numpy.shape(b)) for b in self.__biases]
        B_update = [numpy.zeros(numpy.shape(w)) for w in self.__weights]
        inputs = [x, y]
        # print(inputs)


        hidden_layer, outputs = self.feedForward(inputs)
        print('\n Hidden Layer', hidden_layer)
        # print(inputs,'-->', outputs)

        # outputs[0] = z 
        # 1 - outputs[0] = 1-z
        p = [1, 0.00001]
        q = [1-outputs[0], outputs[0]]
        # q = [0.25, 0.75]

        loss = self.cross_entropy(p, q)
        # Watch
        # BIASES ALWAYS 0 --> FIX
        # https://medium.com/dataseries/back-propagation-algorithm-and-bias-neural-networks-fcf68b5153
        # https://www.askpython.com/python/examples/backpropagation-in-python#:~:text=%20Implementing%20Backpropagation%20in%20Python%20%201%20Import,Split%20Dataset%20in%20Training%20and%20Testing%20More%20
        dW = loss * outputs[0] * (1 - outputs[0]) # THIS IS FOR THE OUTPUT LAYER ONLY 

        # WE NEED DIFFERENT DW FOR INTERMEDIATE LAYERS
        

        # print('Output -> {}\t\tLoss -> {}\t\tdW -> {}'.format(numpy.round(outputs[0], 4), numpy.round(loss, 4), numpy.round(dW, 4)))


        for i in range (1, len(self.__layers)):
            prev_layer = i-1
            B_delta = numpy.dot(numpy.transpose(self.__biases[prev_layer]), loss)
            B_update[prev_layer] = B_delta
        
        # print('\nBIASES', B_update[0])
        # print('\nBIASES', B_update[1])
            
            # print(W_update, '\n')

        for i in range (1, len(self.__layers)):
            # TODO code for output node is different than intermediate node
            Wd_update = numpy.dot(numpy.transpose(self.__weights[i-1]), dW) / batch_size
            W_update[i-1] = numpy.transpose(Wd_update)

        # print('\nWEIGHTS', W_update[0])
        # print('\nWEIGHTS', W_update[1])

        # print('\n',W_update, '\n')
        return (B_update, W_update) 

        



                          
if __name__ == '__main__':
    network = Network([2,20,1])
    train_data = pandas.read_csv('./train_samples_F1.csv')
    # print(len(train_data))
    # print('\n',network.cross_entropy([1, 0.00001], [0.86, 0.14]))
    network.SGD(training_data=train_data, epochs=1, mini_batch_size=10, eta=0.1)
    
    # hidden_layer, outputs = network.feedForward([0.18283921622077468,0.8977168520050608])
    



