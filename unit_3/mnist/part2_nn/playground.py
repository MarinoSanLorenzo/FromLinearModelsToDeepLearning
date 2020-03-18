import numpy as np
try:
	from FromLinearModelsToDeepLearning.unit_3.mnist.part2_nn.neural_nets import rectified_linear_unit, rectified_linear_unit_derivative_vec, NeuralNetwork, relu_prime
except ModuleNotFoundError:
	from neural_nets import rectified_linear_unit, rectified_linear_unit_derivative_vec, NeuralNetwork,  relu_prime

input_values = np.matrix([[-2], [3]])
y2= 8
n = NeuralNetwork()
n.biases = np.matrix('1;2;-8')
n.input_to_hidden_weights = np.matrix(np.arange(-3,3).reshape(3,2))
n.hidden_to_output_weights = np.matrix('1 2 3 ')
layer_weighted_input = n.calculate_layer_weighted_input(input_values)
hidden_layer = n.calc_hidden_layer()
output = n.calc_output()
output_layer_error = n.get_output_layer_error(y2)
hidden_layer_error = n.get_hidden_layer_error()
n.hidden_layer_error
relu_prime(n.hidden_layer)
bias_gradients = n.get_bias_gradients()
hidden_to_output_weights_gradients = n.get_hidden_to_output_weights_gradients()
input_to_hidden_weight_gradients = n.get_input_to_hidden_weights_gradients(input_values)




# test
x1,x2,y = 1,2,3
input_values = np.matrix([[x1],[x2]]) # 2 by 1
n = NeuralNetwork()
hidden_layer_weighted_input = n.calculate_layer_weighted_input(input_values) # TODO (3 by 1 matrix)
hidden_layer_activation = n.calc_hidden_layer()

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

training_pairs = [((2, 1), 10), ((3, 3), 21), ((4, 5), 32), ((6, 6), 42)]
n = NeuralNetwork()
n.training_points = training_pairs
assert n.training_points == training_pairs
# n.train_neural_network()

training_points = iter(n.training_points)
epochs = 0

try:
	x,y = next(training_points)
	n.train(x[0], x[1], y)
	print(f'epochs:\t{epochs}')
	print(n.input_to_hidden_weights)
	print(n.hidden_to_output_weights)
	print(n.biases)
except StopIteration:
	print('done')
	# epochs += 1
	# training_points = iter(n.training_points)

#
# Training pairs:  [((2, 1), 10), ((3, 3), 21), ((4, 5), 32), ((6, 6), 42)]
# Starting params:
#
# (Input --> Hidden Layer) Weights:  [[1. 1.]
#  [1. 1.]
#  [1. 1.]]
# (Hidden --> Output Layer) Weights:  [[1. 1. 1.]]
# Biases:  [[0.]
#  [0.]
#  [0.]]
#
# Epoch  0
# (Input --> Hidden Layer) Weights:  [[1.002 1.001]
#  [1.002 1.001]
#  [1.002 1.001]]
# (Hidden --> Output Layer) Weights:  [[1.003 1.003 1.003]]
# Biases:  [[0.001]
#  [0.001]
#  [0.001]]
# (Input --> Hidden Layer) Weights:  [[1.01077397 1.00977397]
#  [1.01077397 1.00977397]
#  [1.01077397 1.00977397]]
# (Hidden --> Output Layer) Weights:  [[1.02052462 1.02052462 1.02052462]]
# Biases:  [[0.00392466]
#  [0.00392466]
#  [0.00392466]]
# (Input --> Hidden Layer) Weights:  [[1.02772391 1.03096139]
#  [1.02772391 1.03096139]
#  [1.02772391 1.03096139]]
# (Hidden --> Output Layer) Weights:  [[1.05829312 1.05829312 1.05829312]]
# Biases:  [[0.00816214]
#  [0.00816214]
#  [0.00816214]]
# (Input --> Hidden Layer) Weights:  [[1.04523414 1.04847162]
#  [1.04523414 1.04847162]
#  [1.04523414 1.04847162]]
# (Hidden --> Output Layer) Weights:  [[1.09237808 1.09237808 1.09237808]]
# Biases:  [[0.01108051]
#  [0.01108051]
#  [0.01108051]]

from sympy import *
w1,w2, w0, x1, x2 = symbols('w1 w2 w0 x1 x2')
diff(w1*x1+w2*x2 +w0, w0)