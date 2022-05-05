import numpy as np
from tqdm import trange

# np.random.seed(42)

class Layer:
    def __init__(self):
        pass
    
    def forward(self, input):
        pass

class Dense(Layer):
    def __init__(self, inputs, outputs, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # initialize weights with small random numbers. We use normal initialization
        self.weights = np.random.randn(inputs, outputs)*0.01
        self.biases = np.zeros(outputs)
        
    def forward(self,input):
        return np.matmul(input, self.weights) + self.biases
      
    def backward(self,input,grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output,np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        gradient_weights = np.transpose(np.dot(np.transpose(grad_output),input))
        gradient_biases = np.sum(grad_output, axis = 0)
        
        # Here we perform a stochastic gradient descent step. 
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * gradient_weights
        self.biases = self.biases - self.learning_rate * gradient_biases
        return grad_input

class ReLU(Layer):

    def forward(self, input):
        return np.maximum(0,input)

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad

class Sigmoid(Layer):

    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        return grad_output* self.forward(input) * (1 - self.forward(input)) 

def softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    # print(logits)
    # print(reference_answers)
    #logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    
    xentropy = -np.mean(reference_answers*np.log(logits)+(1-reference_answers)*np.log(1-logits))
    
    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]

import numpy as np
from matplotlib import pyplot as plt

F1_OFFSET = 1

TRAIN_SAMPLES = 1_000_000
TEST_SAMPLES  = 200
VAL = 2000

EPOCHS = 50

def normal_pdf(x, loc=0.0, scale=1.0):
    return np.exp(- (((x-loc) / scale) ** 2) / 2) / np.sqrt(2 * np.pi)


X0 = np.random.normal(loc=0.0, scale=1.0, size=(TRAIN_SAMPLES, 2))
X0_TEST = np.random.normal(loc=0.0, scale=1.0, size=(TEST_SAMPLES, 2))

X1 = np.random.normal(loc=0.0, scale=1.0, size=(TRAIN_SAMPLES, 2)) + np.random.choice([-F1_OFFSET, F1_OFFSET], size=(TRAIN_SAMPLES, 2))
X1_TEST = np.random.normal(loc=0.0, scale=1.0, size=(TEST_SAMPLES, 2)) + np.random.choice([-F1_OFFSET, F1_OFFSET], size=(TEST_SAMPLES, 2))
# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)
X_train = np.vstack((X0_TEST, X1_TEST))
y_train = np.vstack((np.zeros((TEST_SAMPLES, 1)), np.ones((TEST_SAMPLES, 1))))

X_val = np.vstack((X0[:VAL], X1[:VAL]))
y_val = np.vstack((np.zeros((VAL, 1)), np.ones((VAL, 1))))

X_test = np.vstack((X0, X1))
y_test = np.vstack((np.zeros((TRAIN_SAMPLES, 1)), np.ones((TRAIN_SAMPLES, 1))))

# plt.figure(figsize=[6,6])
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.title("Label: %i"%y_train[i])
#     plt.imshow(X_train[i].reshape([28,28]),cmap='gray');
    
network = []
network.append(Dense(X_train.shape[1],20))
network.append(ReLU())
network.append(Dense(20, 1))
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
    loss = softmax_crossentropy_with_logits(logits,y)
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
for epoch in range(EPOCHS):

    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=20,shuffle=True):
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

# plt.scatter(X0[:, 0], X0[:, 1], s=2, c="red", marker='.')
# plt.scatter(X1[:, 0], X1[:, 1], s=2, c="green", marker='.')
# plt.show()