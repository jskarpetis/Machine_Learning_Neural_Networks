from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError