import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)

        self.hidden_weights1 = 2 * np.random.random((3, 4)) - 1 
        self.hidden_weights2 = 2 * np.random.random((4, 5)) - 1 
        self.output_weights = 2 * np.random.random((5, 1)) - 1   

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):

            hidden_layer1_activation = self.sigmoid(np.dot(training_inputs, self.hidden_weights1))
            hidden_layer2_activation = self.sigmoid(np.dot(hidden_layer1_activation, self.hidden_weights2))
            output_layer_activation = self.sigmoid(np.dot(hidden_layer2_activation, self.output_weights))

            output_error = training_outputs - output_layer_activation
            output_delta = output_error * self.sigmoid_derivative(output_layer_activation)

            hidden_error2 = output_delta.dot(self.output_weights.T)
            hidden_delta2 = hidden_error2 * self.sigmoid_derivative(hidden_layer2_activation)

            hidden_error1 = hidden_delta2.dot(self.hidden_weights2.T)
            hidden_delta1 = hidden_error1 * self.sigmoid_derivative(hidden_layer1_activation)

            self.output_weights += hidden_layer2_activation.T.dot(output_delta)
            self.hidden_weights2 += hidden_layer1_activation.T.dot(hidden_delta2)
            self.hidden_weights1 += training_inputs.T.dot(hidden_delta1)

    def think(self, inputs):
        inputs = inputs.astype(float)
        hidden_layer1_activation = self.sigmoid(np.dot(inputs, self.hidden_weights1))
        hidden_layer2_activation = self.sigmoid(np.dot(hidden_layer1_activation, self.hidden_weights2))
        output_layer_activation = self.sigmoid(np.dot(hidden_layer2_activation, self.output_weights))
        return output_layer_activation
    

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print('Random Initial Hidden Weights 1:')
    print(neural_network.hidden_weights1)
    print('Random Initial Hidden Weights 2:')
    print(neural_network.hidden_weights2)
    print('Random Initial Output Weights:')
    print(neural_network.output_weights)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    training_outputs = np.array([[0], [1], [1], [0]])

    neural_network.train(training_inputs, training_outputs, 100000)

    print('Hidden Weights 1 after Training:')
    print(neural_network.hidden_weights1)
    print('Hidden Weights 2 after Training:')
    print(neural_network.hidden_weights2)
    print('Output Weights after Training:')
    print(neural_network.output_weights)

    A = float(input('Input 1: '))
    B = float(input('Input 2: '))
    C = float(input('Input 3: '))

    print('New Input:', A, B, C)
    print('Output:', neural_network.think(np.array([A, B, C])))
