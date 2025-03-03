import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)

        # weights for every layer
        self.hidden_weights1 = 2 * np.random.random((3, 4)) - 1  # Input - first hidden layer (3x4)
        self.hidden_weights2 = 2 * np.random.random((4, 5)) - 1  # first hidden layer - second hidden layer (4x5)
        self.output_weights = 2 * np.random.random((5, 1)) - 1   # second hidden layer - output (5x1)

    # formulas and shit
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)  

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            # forward propagation with ReLU in the first hidden layer and sigmoid in the second
            hidden_layer1_activation = self.relu(np.dot(training_inputs, self.hidden_weights1))
            hidden_layer2_activation = self.sigmoid(np.dot(hidden_layer1_activation, self.hidden_weights2))
            output_layer_activation = self.sigmoid(np.dot(hidden_layer2_activation, self.output_weights))

            # error calc
            output_error = training_outputs - output_layer_activation
            output_delta = output_error * self.sigmoid_derivative(output_layer_activation)

            hidden_error2 = output_delta.dot(self.output_weights.T)
            hidden_delta2 = hidden_error2 * self.sigmoid_derivative(hidden_layer2_activation)  

            hidden_error1 = hidden_delta2.dot(self.hidden_weights2.T)
            hidden_delta1 = hidden_error1 * self.relu_derivative(hidden_layer1_activation) 

            # updating wegihts
            self.output_weights += hidden_layer2_activation.T.dot(output_delta)
            self.hidden_weights2 += hidden_layer1_activation.T.dot(hidden_delta2)
            self.hidden_weights1 += training_inputs.T.dot(hidden_delta1)

    def think(self, inputs):
        inputs = inputs.astype(float)
        hidden_layer1_activation = self.relu(np.dot(inputs, self.hidden_weights1))
        hidden_layer2_activation = self.sigmoid(np.dot(hidden_layer1_activation, self.hidden_weights2))
        output_layer_activation = self.sigmoid(np.dot(hidden_layer2_activation, self.output_weights))
        return output_layer_activation
    
# text
if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print('Random Initial Hidden Weights 1:')
    print(neural_network.hidden_weights1)
    print('Random Initial Hidden Weights 2:')
    print(neural_network.hidden_weights2)
    print('Random Initial Output Weights:')
    print(neural_network.output_weights)

    # input
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1],
                                [0, 0, 0]])

    training_outputs = np.array([[0], [1], [1], [0], [0]])

    neural_network.train(training_inputs, training_outputs, 500000)

    #text
    print('Hidden Weights 1 after Training:')
    print(neural_network.hidden_weights1)
    print('Hidden Weights 2 after Training:')
    print(neural_network.hidden_weights2)
    print('Output Weights after Training:')
    print(neural_network.output_weights)
    #input
    A = float(input('Input 1: '))
    B = float(input('Input 2: '))
    C = float(input('Input 3: '))
    #output
    print('New Input:', A, B, C)
    print('Output:', neural_network.think(np.array([A, B, C])))
    input('')
