from numpy import *

class functions:

    # init network, set weights to 0
    def __init__(self, inputs, targets, hidden_layer_size, hidden_layer2):

        self.I = inputs
        self.T = targets
        self.H_layer_size = hidden_layer_size
        self.H_2 = hidden_layer2
        self.I_layer_size = shape(inputs)[1]
        self.O_layer_size = shape(targets)[1]
        self.examples = shape(inputs)[0]

        self.I = concatenate((self.I, -ones((self.examples, 1))), axis=1)
        self.H, self.H2, self.O, self.W1, self.W2, self.W3, self.dW1, self.dW2, self.dW3  = None, None, None, None, None, None, None, None, None

    # set weights to random values between -1 and 1, init update weights to 0
    def randomize(self):    
        self.W1 = (random.rand(self.I_layer_size + 1, self.H_layer_size) - 0.5) * 2 / sqrt(self.I_layer_size)
        self.W2 = (random.rand(self.H_layer_size + 1, self.H_2) - 0.5) * 2 / sqrt(self.H_layer_size)
        self.W3 = (random.rand(self.H_2 + 1, self.O_layer_size) - 0.5) * 2 / sqrt(self.H_2)
        self.dW1 = zeros((shape(self.W1)))
        self.dW2 = zeros((shape(self.W2)))
        self.dW3 = zeros((shape(self.W3)))


    # feed input set through network, calculate results from weights at each node
    def forward(self, inputs):    
        self.H = dot(inputs, self.W1)
        self.H = 1.0 / (1.0 + exp(-self.H))
        self.H = concatenate((self.H, -ones((shape(inputs)[0], 1))), axis=1)

        self.H2 = dot(self.H, self.W2)
        self.H2 = 1.0 / (1.0 + exp(-self.H2))
        self.H2 = concatenate((self.H2, -ones((shape(self.H)[0], 1))), axis=1)

        self.O = dot(self.H2, self.W3)

        return self.O

        
    # calculate values for backprop, find error and weight changes
    def backprop(self):
        Z2 = (self.O - self.T) / self.examples
        Z1 = self.H * (1.0 - self.H) * (dot(Z2, transpose(self.W2)))
        self.dW1 = dot(transpose(self.I), Z1[:, :-1])
        self.dW2 = dot(transpose(self.H), Z2)
        self.dW3 = dot(transpose(self.H2), Z2)

        error = 0.5 * sum((self.O - self.T) ** 2)

        return error

    # updates weights based on backprop calculation
    def update_weights(self, lrate):
        self.W1 -= lrate * self.dW1
        self.W2 -= lrate * self.dW2
        self.W3 -= lrate * self.dW3