import numpy
import math

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
        
        print('\n')
        print(self.__neurons)
    
    def __initBiases(self):
        self.__biases = [ [float(0) for _ in range(self.__layers[i])] for i in range(1, len(self.__layers)) ]
        print('\n')
        print(self.__biases)
    
    def __initWeights(self):
        self.__weights = [[[float(self.generateRandomNumber(self.__layers[i-1], self.__layers[i])) for _ in range(self.__layers[i-1])] for _ in range(self.__layers[i])] for i in range(1, len(self.__layers))]
        print('\n')
        print(self.__weights)
        
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


if __name__ == '__main__':
    network = Network([2,20,1])
    network.feedForward([1.04234, -0.524])
    
    



