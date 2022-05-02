import random 

class Network :
    __layers = None
    __neurons = None
    __biases = None
    __weights = None

    fitness: float = 0

    def __init__(self,layers):

        self.__layers = [layers[i] for i in range(self.layers)]

        self.__initNeurons()
        self.initBiases()
        self.initWeights()
    
    def __initNeurons(self) :
        self.__neurons = [ float(self.__layers[i]) for i in range(len(self.__layers)) ]
    



