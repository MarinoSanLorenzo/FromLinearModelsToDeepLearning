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
	def tearDown(self):
		print('Running tear down...')

	def test_rectified_linear_unit(self):
		is_true = np.allclose(rectified_linear_unit(self.x), np.array([0, 0, 0, 0, 1, 2]))
		self.assertTrue(is_true)

	def test_relu_dev(self):
		is_true = np.allclose(rectified_linear_unit_derivative_vec(self.x), np.array([0, 0, 0, 0, 1, 1]))
		self.assertTrue(is_true)

	def test_calc_hidden_layer_ok(self):
		n = NeuralNetwork()
		self.assertTrue(np.allclose(np.matrix([[3], [3], [3]]), n.calculate_layer_weighted_input(self.input_values)))

		hidden_layer = rectified_linear_unit(n.calculate_layer_weighted_input(self.input_values))
		r, c = hidden_layer.shape
		self.assertEqual(r,3)
		self.assertEqual(c, 1)

if __name__ == '__main__':
    run_tests(TestNeuralNetwork)