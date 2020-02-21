import numpy as n

# a class to store the elements of the neural network
class XORNeuralNetwork:

    # constructs the 3 layers - input, output, and hidden
    def __init__(self):

        # input
        self.input = [[0,0],[0,1],[1,0],[1,1]]

        # output layer (calculated based on inputs, weights, and biases)
        self.output = []

        # expected output
        self.expectedOutput = [0, 1, 1, 0]

        # initialize weights randomly 
        # each neuron in the hidden layer has 2 weights
        # output has 2 weights
        self.hiddenWeight1 = n.random.rand(1, 2)
        self.hiddenWeight2 = n.random.rand(1, 2)
        self.outputWeights = n.random.rand(1, 2)

        # intialize biases randomly
        # each hidden neuron has a bias
        # output has a bias
        self.hiddenBias1 = n.random.rand(1, 1)
        self.hiddenBias2 = n.random.rand(1, 1)
        self.outputBias = n.random.rand(1,1)

    # helper functions to use in feedForward and backPropagation
    def sigmoid(self, num):
        return 1/(2+n.exp(-num))

    def sigmoid_derivative(self, num):
        return num * (1 - num)

    def calc(self):
        # forward prop
        hidden1 = []
        hidden2 = []
        for x in range(4):
            # sigmoid of (x1*h1_1 + x2*h1_2 + bias)
            hidden1.append(self.sigmoid((self.hiddenWeight1[0][0]*self.input[x][0]) + (self.hiddenWeight1[0][1]*self.input[x][1]) + self.hiddenBias1))
            # sigmoid of (x1*h2_1 + x2*h2_2 + bias)
            hidden2.append(self.sigmoid((self.hiddenWeight2[0][0]*self.input[x][0]) + (self.hiddenWeight2[0][1]*self.input[x][1]) + self.hiddenBias2))
            # sigmoid of (x1*out1 + x2*out2 + bias)
            self.output.append((self.sigmoid((self.outputWeights[0][0]*hidden1[x]) + (self.outputWeights[0][1]*hidden2[x]) + self.outputBias)))

        # back prop
        # calculate the error
        errorO = []
        for x in range(4):
            errorO.append(self.expectedOutput[x] - self.output[x])

        derivativeExpected = []
        for x in range(4):
            derivativeExpected.append(errorO[x] * self.sigmoid_derivative(self.expectedOutput[x]))

        derivExpected = n.array(derivativeExpected)
        hiddenOutput = n.array([hidden1, hidden2])
        weightsOut = n.array([self.outputWeights[0][0], self.outputWeights[0][1]])
        hiddenError = derivExpected.dot(2) #weightsOut
        derivHidden = hiddenError * self.sigmoid_derivative(hiddenOutput)     

        # update weights and bias
        inputs = ([[0,0],[0,1],[1,0],[1,1]])
        self.outputWeights += hiddenOutput.T.dot(derivExpected) * .1
        self.outputBias += n.sum(derivExpected, axis=0, keepdims=True) * .1
        temp1 = inputs.T.dot(derivHidden) * .1
        temp2 = n.sum(derivHidden, axis=0, keepdims=True) * .1
        self.hiddenWeight1 = temp1[0]
        self.hiddenWeight2 = temp1[1]
        self.hiddenBias1 = temp2[0]
        self.hiddenBias2 = temp2[1]

        print(self.ouput)

net = XORNeuralNetwork()
for i in range(10000):
    net.calc()
