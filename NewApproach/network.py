import random
import numpy as np
from pyrsistent import b
import pandas 
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

#Derivative of sigmoid
def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

class Network:
    # sizes is a list of the number of nodes in each layer. e.x [2 20 1]
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.array([0 for _ in range(y)]) for y in sizes[1:]]
        self.weights = [ (1/(m+n))**(1/2) + np.random.randn(m,n) for n,m in zip(sizes[:-1], sizes[1:])]
    
    def __str__(self):
        return f"Layers of Network {self.num_layers}\nShape of Network {self.sizes}\nBiases {self.biases}\nWeights {self.weights}"

    def ff(self,activation):
        for bias,weight in zip(self.biases,self.weights):
            activation = sigmoid(np.dot(weight,activation) + bias) # σ(w*a + b)
        return activation

    def exponential_loss(self,yhat: np.ndarray, y: np.ndarray):
        return np.mean(np.exp(-0.5*yhat*y) + np.exp(0.5*yhat*y))

    def binary_cross_entropy(self,yhat: np.ndarray, y: np.ndarray):
        return -(y * np.log(yhat) + (1-y) * np.log(1 - yhat)).mean()

    #Stohastic Gradient Descent
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        training_data = list(training_data)
        samples = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, samples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {epoch} complete")
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # stores activations layer by layer
        zs = [] # stores z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
       
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for _layer in range(2, self.num_layers):
            z = zs[-_layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-_layer+1].transpose(), delta) * sp
            nabla_b[-_layer] = delta
            nabla_w[-_layer] = np.dot(delta, activations[-_layer-1].transpose())
        return (nabla_b, nabla_w)
    
    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b-(learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    

if __name__=="__main__":
    net = Network([2,20,1])
    train_data = pandas.read_csv('./train_samples_F1.csv').to_numpy()
    print(net)
    # print(len(train_data))
    # print('\n',network.cross_entropy([1, 0.00001], [0.86, 0.14]))
    net.SGD(training_data=train_data, epochs=1, mini_batch_size=10, learning_rate=0.01)
