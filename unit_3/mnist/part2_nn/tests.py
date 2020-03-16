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
		self.x = np.arange(-3,3)
		self.input_values = np.matrix([[1], [2]])
		self.n = NeuralNetwork()
		self.hidden_layer = self.n.calc_hidden_layer(self.input_values)
		self.output = self.n.calc_output(self.hidden_layer)
		self.y = 9
		self.output_layer_error = self.n.get_output_layer_error(self.y)
		self.hidden_layer_error = self.n.get_hidden_layer_error()
		self.bias_gradients = self.n.get_bias_gradients()
		self.hidden_to_output_weights_gradients = self.n.get_hidden_to_output_weigts_gradients()
		self.input_to_hidden_weight_gradients = self.n.get_input_to_hidden_weight_gradients(self.input_values)
	def tearDown(self):
		print('Running tear down...')

	def test_rectified_linear_unit(self):
		is_true = np.allclose(rectified_linear_unit(self.x), np.array([0, 0, 0, 0, 1, 2]))
		self.assertTrue(is_true)

	def test_relu_dev(self):
		is_true = np.allclose(rectified_linear_unit_derivative_vec(self.x), np.array([0, 0, 0, 0, 1, 1]))
		self.assertTrue(is_true)

	def test_calc_hidden_layer_ok(self):
		self.assertTrue(np.allclose(np.matrix([[3], [3], [3]]), self.n.calculate_layer_weighted_input(self.input_values)))

		r, c = self.hidden_layer.shape
		self.assertEqual(r,3)
		self.assertEqual(c, 1)
		self.assertTrue(np.allclose(np.matrix([[3], [3], [3]]), self.hidden_layer))

	def test_calc_output_ok(self):
		self.assertEqual(self.y, self.output)

	def test_output_layer_error(self):
		self.assertEqual(0, self.output_layer_error)

	def test_hidden_layer_error(self):
		r,c = self.hidden_layer_error.shape
		self.assertEqual(3,r)
		self.assertEqual(1, c)

		self.assertTrue(np.allclose(np.matrix([[0],[0],[0]]), self.hidden_layer_error ))

	def test_bias_gradient(self):
		self.assertTrue(np.allclose(np.matrix([[0],[0],[0]]), self.bias_gradients ))

	def test_hidden_to_output_weight_gradients(self):
		r,c = self.hidden_to_output_weights_gradients.shape
		self.assertEqual(1, r)
		self.assertEqual(3, c)

		self.assertTrue(np.allclose(np.matrix([[0], [0], [0]]), self.hidden_to_output_weights_gradients))

	def test_input_to_hidden_weight_gradients(self):
		r,c = self.input_to_hidden_weight_gradients.shape
		self.assertEqual(3, r)
		self.assertEqual(2, c)

		self.assertTrue(np.allclose(np.repeat(0,6).reshape(3,2), self.input_to_hidden_weight_gradients))


if __name__ == '__main__':
    run_tests(TestNeuralNetwork)