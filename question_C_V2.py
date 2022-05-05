from inspect import getfile
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
import os
EPOCHS = 15
MINIBATCH_SIZE = 20

origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
path = getfile(
    "mnist data/",
    origin=origin_folder + 'mnist.npz',
    file_hash=
    '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')

def load_mnist(path):

    with np.load(path, allow_pickle=True) as f:  # pylint: disable=unexpected-keyword-arg
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

(train_X_all, train_y_all), (test_X_all, test_y_all) = load_mnist("mnist data/mnist.npz")

def filter_08s(Xs, ys):
    clean_x = []
    clean_y = []

    for X, y in zip(Xs, ys):
        if y in (0, 8):
            clean_x.append(X.reshape((784,)) / 255)
            clean_y.append(0 if y == 0 else 1)
        
    return np.asarray(clean_x), np.asarray(clean_y).reshape((len(clean_y), 1))


X_train, y_train = filter_08s(train_X_all, train_y_all)
X_val, y_val = filter_08s(test_X_all, test_y_all)

# print(train_X[0], train_y[0])
class Layer:
    def __init__(self):
        pass
    
    def forward(self, input):
        pass

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # initialize weights with small random numbers. We use normal initialization
        self.weights = np.random.randn(input_units, output_units)*0.01
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        return np.matmul(input, self.weights) + self.biases
      
    def backward(self,input,grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output,np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        grad_weights = np.transpose(np.dot(np.transpose(grad_output),input))
        grad_biases = np.sum(grad_output, axis = 0)
        
        # Here we perform a stochastic gradient descent step. 
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input

class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        pass
    
    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return np.maximum(0,input)

    def backward(self, input, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output*relu_grad

class Sigmoid(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return 1 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        return grad_output* self.forward(input) * (1 - self.forward(input)) 

def crossentropy(logits, reference_answers):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    xentropy = -np.mean(reference_answers*np.log(logits)+(1-reference_answers)*np.log(1-logits))
    
    return xentropy

def grad_crossentropy(logits,reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]


network = []
network.append(Dense(784, 300))
network.append(ReLU())
network.append(Dense(300, 1))
network.append(Sigmoid())
# network.append(ReLU())
# network.append(Dense(200,10))

def forward(network, X):
    """
    Compute activations of all network layers by applying them sequentially.
    Return a list of activations for each layer. 
    Make sure last activation corresponds to network logits.
    """
    activations = []
    input = X
    for i in range(len(network)):
        activations.append(network[i].forward(X))
        X = network[i].forward(X)
        
    assert len(activations) == len(network)
    return activations

def predict(network,X):
    """
    Compute network predictions.
    """
    logits = forward(network,X)[-1]
    return np.heaviside(logits-0.5, 0)

def train(network,X,y):
    """
    Train your network on a given batch of X and y.
    You first need to run forward to get all layer activations.
    Then you can run layer.backward going from last to first layer.
    After you called backward for all layers, all Dense layers have already made one gradient step.
    """
    
    # Get the layer activations
    layer_activations = forward(network,X)
    logits = layer_activations[-1]
    
    # Compute the loss and the initial gradient
    loss = crossentropy(logits,y)
    loss_grad = logits - y #loss #grad_softmax_crossentropy_with_logits(logits,y)
    
    for i in range(1, len(network)):
        loss_grad = network[len(network) - i].backward(layer_activations[len(network) - i - 1], loss_grad)
    
    return np.mean(loss)
  

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
train_log = []
val_log = []

train_log.append(np.mean(predict(network,X_train)==y_train))
val_log.append(np.mean(predict(network,X_val)==y_val))

# clear_output()
print("Uninitialized", 0)
print("Train accuracy:",train_log[-1])
print("Val accuracy:",val_log[-1])

for epoch in range(EPOCHS):

    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=MINIBATCH_SIZE,shuffle=True):
        train(network,x_batch,y_batch)
    
    train_log.append(np.mean(predict(network,X_train)==y_train))
    val_log.append(np.mean(predict(network,X_val)==y_val))
    
    # clear_output()
    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    print("Val accuracy:",val_log[-1])

plt.plot(train_log,label='train accuracy')
plt.plot(val_log,label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()