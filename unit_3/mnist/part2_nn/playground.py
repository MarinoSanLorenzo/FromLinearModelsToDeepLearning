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
layer_weighted_input = n.calculate_layer_weighted_input(input_values)
hidden_layer = n.calc_hidden_layer()
n.hidden_to_output_weights = np.matrix('1 2 3 ')
output = n.calc_output()
output_layer_error = n.get_output_layer_error(y2)
hidden_layer_error = n.get_hidden_layer_error()
bias_gradients = n.get_bias_gradients()
hidden_to_output_weights_gradients = n.get_hidden_to_output_weights_gradients()
exp = np.matrix([[-1], [-4], [0]])
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

training_pairs = [((-2, 3), 1), ((-5, 7), 2), ((-6, -8), -14), ((0, 6), 6), ((-1, 8), 7), ((-8, 1), -7), ((6, 7), 13),
                  ((-10, 6), -4), ((-8, -6), -14), ((-6, 8), 2)]

n =  NeuralNetwork()
n.training_points = training_pairs
assert n.training_points == training_pairs
n.train_neural_network()

# Epoch  0
# (Input --> Hidden Layer) Weights:  [[0.9267717  0.56478874]
#  [0.9267717  0.56478874]
#  [0.9267717  0.56478874]]
# (Hidden --> Output Layer) Weights:  [[0.47777857 0.47777857 0.47777857]]
# Biases:  [[-0.06567411]
#  [-0.06567411]
#  [-0.06567411]]
# Epoch  1
# (Input --> Hidden Layer) Weights:  [[0.91700237 0.57629692]
#  [0.91700237 0.57629692]
#  [0.91700237 0.57629692]]
# (Hidden --> Output Layer) Weights:  [[0.47294759 0.47294759 0.47294759]]
# Biases:  [[-0.06410768]
#  [-0.06410768]
#  [-0.06410768]]
# Epoch  2
# (Input --> Hidden Layer) Weights:  [[0.90819495 0.58759469]
#  [0.90819495 0.58759469]
#  [0.90819495 0.58759469]]
# (Hidden --> Output Layer) Weights:  [[0.47008217 0.47008217 0.47008217]]
# Biases:  [[-0.062565]
#  [-0.062565]
#  [-0.062565]]
# Epoch  3
# (Input --> Hidden Layer) Weights:  [[0.90000188 0.59816995]
#  [0.90000188 0.59816995]
#  [0.90000188 0.59816995]]
# (Hidden --> Output Layer) Weights:  [[0.46787518 0.46787518 0.46787518]]
# Biases:  [[-0.06111995]
#  [-0.06111995]
#  [-0.06111995]]
# Epoch  4
# (Input --> Hidden Layer) Weights:  [[0.8923118  0.60796852]
#  [0.8923118  0.60796852]
#  [0.8923118  0.60796852]]
# (Hidden --> Output Layer) Weights:  [[0.46597909 0.46597909 0.46597909]]
# Biases:  [[-0.05978086]
#  [-0.05978086]
#  [-0.05978086]]
# Epoch  5
# (Input --> Hidden Layer) Weights:  [[0.88507271 0.61703822]
#  [0.88507271 0.61703822]
#  [0.88507271 0.61703822]]
# (Hidden --> Output Layer) Weights:  [[0.46428854 0.46428854 0.46428854]]
# Biases:  [[-0.05854136]
#  [-0.05854136]
#  [-0.05854136]]
# Epoch  6
# (Input --> Hidden Layer) Weights:  [[0.87647937 0.62776711]
#  [0.87647937 0.62776711]
#  [0.87647937 0.62776711]]
# (Hidden --> Output Layer) Weights:  [[0.46243638 0.46243638 0.46243638]]
# Biases:  [[-0.05654563]
#  [-0.05654563]
#  [-0.05654563]]
# Epoch  7
# (Input --> Hidden Layer) Weights:  [[0.8684848  0.63748698]
#  [0.8684848  0.63748698]
#  [0.8684848  0.63748698]]
# (Hidden --> Output Layer) Weights:  [[0.46071941 0.46071941 0.46071941]]
# Biases:  [[-0.0547291]
#  [-0.0547291]
#  [-0.0547291]]
# Epoch  8
# (Input --> Hidden Layer) Weights:  [[0.84225541 0.6706392 ]
#  [0.84225541 0.6706392 ]
#  [0.84225541 0.6706392 ]]
# (Hidden --> Output Layer) Weights:  [[0.45821565 0.45821565 0.45821565]]
# Biases:  [[-0.049838]
#  [-0.049838]
#  [-0.049838]]
# Epoch  9
# (Input --> Hidden Layer) Weights:  [[0.82232568 0.69343711]
#  [0.82232568 0.69343711]
#  [0.82232568 0.69343711]]
# (Hidden --> Output Layer) Weights:  [[0.45548505 0.45548505 0.45548505]]
# Biases:  [[-0.04646597]
#  [-0.04646597]
#  [-0.04646597]]