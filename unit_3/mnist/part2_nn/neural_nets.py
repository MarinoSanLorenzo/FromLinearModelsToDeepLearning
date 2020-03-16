import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    def max_(x):
        return max(x,0)
    return np.vectorize(max_)(x)

def rectified_linear_unit_derivative_vec(x):
    """ Returns the derivative of ReLU."""
    new_x = rectified_linear_unit(x)
    def relu_dev(new_x):
        if new_x<=0:
            return 0
        elif new_x>0:
            return 1
        else:
            raise NotImplementedError
    return np.vectorize(relu_dev)(new_x)

def rectified_linear_unit_derivative(x):
    z = max(x,0)
    if z <= 0:
        return 0
    elif z > 0:
        return 1
    else:
        raise NotImplementedError

def relu_prime(x):
    def relu_prime_helper(x):
        if x<=0:
            return 0
        return 1
    return np.vectorize(relu_prime_helper)(x)

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
        self.hidden_to_output_weights = np.matrix('1 1 1')
        self.biases = np.matrix('0; 0; 0')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]


    def calculate_layer_weighted_input(self, input_values):
        return self.input_to_hidden_weights*input_values + self.biases

    def calc_hidden_layer(self, input_values):
        # TODO: cash the variables
        hidden_layer_weighted_input = self.calculate_layer_weighted_input(input_values)
        self.hidden_layer = rectified_linear_unit(hidden_layer_weighted_input)
        return self.hidden_layer

    def calc_output(self,calc_hidden_layer, act_func = lambda x: x ):
        self.output = float(act_func(self.hidden_to_output_weights*calc_hidden_layer))
        return self.output

    def get_output_layer_error(self,y):
        self.output_layer_error = (y-self.output)
        return self.output_layer_error
        # self.output_layer_error = (y-self.output)*n.hidden_to_output_weights.transpose( )
        # return self.output_layer_error

    def get_hidden_layer_error(self):
        self.hidden_layer_error = (self.output_layer_error*self.hidden_to_output_weights).transpose()
        return self.hidden_layer_error

    def get_bias_gradients(self):
        self.bias_gradients = np.array(self.hidden_layer_error)*relu_prime(np.array(self.biases))
        return self.bias_gradients

    def get_hidden_to_output_weigts_gradients(self):
        self.hidden_to_output_weights_gradients = self.output_layer_error*self.hidden_layer.transpose()
        return self.hidden_to_output_weights_gradients

    def get_input_to_hidden_weight_gradients(self, input_values):
        prime_hidden_layer_weight = relu_prime(np.concatenate((input_values, input_values, input_values)).reshape(3,2))
        self.input_to_hidden_weight_gradients = np.array(self.hidden_layer_error) * np.array(prime_hidden_layer_weight)
        return self.input_to_hidden_weight_gradients


    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = self.calculate_layer_weighted_input(input_values) # TODO (3 by 1 matrix)
        hidden_layer_activation = self.calc_hidden_layer(input_values)


        output = self.calc_output(hidden_layer_activation) # TODO
        activated_output = self.calc_output(hidden_layer_activation, act_func = lambda x: x )# TODO

        # ### Backpropagation ###

        # Compute gradients
        output_layer_error = self.get_output_layer_error(y) # TODO
        hidden_layer_error = self.get_hidden_layer_error() # TODO (3 by 1 matrix)

        bias_gradients = self.get_bias_gradients() # TODO
        hidden_to_output_weight_gradients = self.get_hidden_to_output_weigts_gradients() # TODO
        input_to_hidden_weight_gradients = self.get_input_to_hidden_weight_gradients(input_values) # TODO

        # # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases- self.learning_rate*bias_gradients # TODO
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate*input_to_hidden_weight_gradients # TODO
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate*hidden_to_output_weight_gradients # TODO

    # def predict(self, x1, x2):
    #
    #     input_values = np.matrix([[x1],[x2]])
    #
    #     # Compute output for a single input(should be same as the forward propagation in training)
    #     hidden_layer_weighted_input = # TODO
    #     hidden_layer_activation = # TODO
    #     output = # TODO
    #     activated_output = # TODO
    #
    #     return activated_output.item()
    #
    # # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        print('Starting params:')
        print(f'biases:\n{self.biases}')
        print(f'input --- > hidden:\n{self.input_to_hidden_weights}')
        print(f'hidden --- > output:\n{self.hidden_to_output_weights}')

        for epoch in range(self.epochs_to_train):
            print(f'epoch:\t{epoch}')
            print('weights:')
            print(f'biases:\n{self.biases}')
            print(f'input --- > hidden:\n{self.input_to_hidden_weights}')
            print(f'hidden --- > output:\n{self.hidden_to_output_weights}')
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # # Run this to test your neural network implementation for correctness after it is trained
    # def test_neural_network(self):
    #
    #     for point in self.testing_points:
    #         print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
    #         if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
    #             print("Test Passed")
    #         else:
    #             print("Point ", point[0], point[1], " failed to be predicted correctly.")
    #             return

# x = NeuralNetwork()

# x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
# x.test_neural_network()
import numpy as np

input_values = np.matrix([[1], [2]])
n = NeuralNetwork()
n.train(1,1,3)
hidden_layer = n.calc_hidden_layer(input_values)
output = n.calc_output(hidden_layer)
y = 9
output_layer_error = n.get_output_layer_error(y)
hidden_layer_error = n.get_hidden_layer_error()
bias_gradient = n.get_bias_gradients()
hidden_to_output_weights_gradients = n.get_hidden_to_output_weigts_gradients()

# test
x1,x2,y = 1,2,3
input_values = np.matrix([[x1],[x2]]) # 2 by 1
n = NeuralNetwork()
hidden_layer_weighted_input = n.calculate_layer_weighted_input(input_values) # TODO (3 by 1 matrix)
hidden_layer_activation = n.calc_hidden_layer(input_values)

output = n.calc_output(hidden_layer_activation) # TODO
activated_output = n.calc_output(hidden_layer_activation, act_func = lambda x: x )# TODO

output_layer_error = n.get_output_layer_error(y) # TODO
hidden_layer_error = n.get_hidden_layer_error() # TODO (3 by 1 matrix)

bias_gradients = n.get_bias_gradients() # TODO
hidden_to_output_weight_gradients = n.get_hidden_to_output_weigts_gradients() # TODO
input_to_hidden_weight_gradients = n.get_input_to_hidden_weight_gradients(input_values) # TODO

# # Use gradients to adjust weights and biases using gradient descent

n.biases = n.biases- n.learning_rate*bias_gradients # TODO
n.input_to_hidden_weights = n.input_to_hidden_weights - n.learning_rate*input_to_hidden_weight_gradients # TODO
n.hidden_to_output_weights = n.hidden_to_output_weights - n.learning_rate*hidden_to_output_weight_gradients # TODO

# tests 2

training_pairs = [((-2, 3), 1), ((-5, 7), 2), ((-6, -8), -14), ((0, 6), 6), ((-1, 8), 7), ((-8, 1), -7), ((6, 7), 13),
                  ((-10, 6), -4), ((-8, -6), -14), ((-6, 8), 2)]

n =  NeuralNetwork()
n.training_points = training_pairs
n.train_neural_network()