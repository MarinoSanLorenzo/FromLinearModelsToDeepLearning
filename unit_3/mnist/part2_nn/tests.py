import numpy as np
import unittest
try:
	from FromLinearModelsToDeepLearning.unit_3.mnist.part2_nn.neural_nets import rectified_linear_unit, rectified_linear_unit_derivative_vec, NeuralNetwork
except ModuleNotFoundError:
	from neural_nets import rectified_linear_unit, rectified_linear_unit_derivative_vec, NeuralNetwork

def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

class TestNeuralNetwork(unittest.TestCase):
	def setUp(self):
		import numpy as np
		# test 1
		self.x = np.arange(-3,3)
		self.input_values = np.matrix([[1], [2]])
		self.n = NeuralNetwork()
		self.layer_weighted_input = self.n.calculate_layer_weighted_input(self.input_values)
		self.hidden_layer = self.n.calc_hidden_layer()
		self.output = self.n.calc_output()
		self.y = 9
		self.output_layer_error = self.n.get_output_layer_error(self.y)
		self.hidden_layer_error = self.n.get_hidden_layer_error()
		self.bias_gradients = self.n.get_bias_gradients()
		# self.hidden_to_output_weights_gradients = self.n.get_hidden_to_output_weigts_gradients()
		# self.input_to_hidden_weight_gradients = self.n.get_input_to_hidden_weight_gradients(self.input_values)

		# test 2

		self.x2 = np.arange(-2,3)
		self.input_values2 = np.matrix([[-2], [3]])
		self.n2 = NeuralNetwork()
		self.n2.biases = np.matrix('1;2;-8')
		self.n2.input_to_hidden_weights = np.matrix(np.arange(-3, 3).reshape(3, 2))
		self.layer_weighted_input2 = self.n2.calculate_layer_weighted_input(self.input_values2)

		self.hidden_layer2 = self.n2.calc_hidden_layer()
		self.n2.hidden_to_output_weights = np.matrix('1 2 3 ')
		self.output2 = self.n2.calc_output()
		self.y2 = 8
		self.output_layer_error2 = self.n2.get_output_layer_error(self.y2)
		self.hidden_layer_error2 = self.n2.get_hidden_layer_error()
		self.bias_gradients2 = self.n2.get_bias_gradients()
		# self.hidden_to_output_weights_gradients2 = self.n2.get_hidden_to_output_weigts_gradients()
		# self.input_to_hidden_weight_gradients2 = self.n2.get_input_to_hidden_weight_gradients(self.input_values2)

	def tearDown(self):
		print('Running tear down...')

	def test_rectified_linear_unit(self):
		is_true = np.allclose(rectified_linear_unit(self.x), np.array([0, 0, 0, 0, 1, 2]))
		self.assertTrue(is_true)

		is_true = np.allclose(rectified_linear_unit(self.x2), np.array([0, 0, 0, 1, 2]))
		self.assertTrue(is_true)

	def test_relu_dev(self):
		is_true = np.allclose(rectified_linear_unit_derivative_vec(self.x), np.array([0, 0, 0, 0, 1, 1]))
		self.assertTrue(is_true)

		is_true = np.allclose(rectified_linear_unit_derivative_vec(self.x2), np.array([ 0, 0, 0, 1, 1]))
		self.assertTrue(is_true)

	def test_calc_hidden_layer_ok(self):
		#test1
		self.assertTrue(np.allclose(np.matrix([[3], [3], [3]]), self.n.calculate_layer_weighted_input(self.input_values)))

		r, c = self.hidden_layer.shape
		self.assertEqual(r,3)
		self.assertEqual(c, 1)
		self.assertTrue(np.allclose(np.matrix([[3], [3], [3]]), self.hidden_layer))

		#test2
		r, c = self.layer_weighted_input2.shape
		self.assertEqual(r, 3)
		self.assertEqual(c, 1)
		self.assertTrue(np.allclose(np.matrix([[1], [4], [-4]]), self.n2.calculate_layer_weighted_input(self.input_values2)))

		r, c = self.hidden_layer2.shape
		self.assertEqual(r,3)
		self.assertEqual(c, 1)
		self.assertTrue(np.allclose(np.matrix([[1], [4], [0]]), self.hidden_layer2))

	def test_calc_output_ok(self):
		self.assertEqual(self.y, self.output)
		self.assertEqual(9, self.output2)

	def test_output_layer_error(self):
		self.assertEqual(0, self.output_layer_error)
		#
		self.assertEqual(-1, self.output_layer_error2)

		# # test1
		# r,c = self.output_layer_error.shape
		# self.assertEqual(3,r)
		# self.assertEqual(1, c)
		#
		# self.assertTrue(np.allclose(np.matrix([[0],[0],[0]]), self.output_layer_error ))
		# # test2
		# r, c = self.output_layer_error2.shape
		# self.assertEqual(3, r)
		# self.assertEqual(1, c)
		#
		# self.assertTrue(np.allclose(np.matrix([[-1], [-2], [-3]]), self.output_layer_error2))

	def test_hidden_layer_error(self):
		# test 1
		r,c = self.hidden_layer_error.shape
		self.assertEqual(3,r)
		self.assertEqual(1, c)

		self.assertTrue(np.allclose(np.matrix([[0],[0],[0]]), self.hidden_layer_error ))
		# test 2
		r, c = self.hidden_layer_error2.shape
		self.assertEqual(3, r)
		self.assertEqual(1, c)

		self.assertTrue(np.allclose(np.matrix([[-1], [-2], [-3]]), self.hidden_layer_error2))

	def test_bias_gradient(self):
		# test 1
		self.assertTrue(np.allclose(np.matrix([[0],[0],[0]]), self.bias_gradients ))
		# test 2
		self.assertTrue(np.allclose(np.matrix([[-1],[-2],[0]]), self.bias_gradients2 ))

	# def test_hidden_to_output_weight_gradients(self):
	# 	r,c = self.hidden_to_output_weights_gradients.shape
	# 	self.assertEqual(1, r)
	# 	self.assertEqual(3, c)
	#
	# 	self.assertTrue(np.allclose(np.matrix([[0], [0], [0]]), self.hidden_to_output_weights_gradients))
	#
	# def test_input_to_hidden_weight_gradients(self):
	# 	r,c = self.input_to_hidden_weight_gradients.shape
	# 	self.assertEqual(3, r)
	# 	self.assertEqual(2, c)
	#
	# 	self.assertTrue(np.allclose(np.repeat(0,6).reshape(3,2), self.input_to_hidden_weight_gradients))


if __name__ == '__main__':
    run_tests(TestNeuralNetwork)