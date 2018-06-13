import numpy as np

class Activation_Function(object):
	def __init__(self, function):
		"""
		Paramters:
		----------
		Function: str
			Selects which activcation function to use.
			'Logostic' - For the logistic function 1/(1 + exp(-x))

		Returns:
		--------
		None
		"""
		self._function = function

	def apply(self, x):
		if self.function == 'logistic':
			return self._logistic(x)
		else:
			raise('Please input a valid activation function.')


	def _logistic(self, x):
		"""
		This is an implementation of the logistic function.
		Also known as the Sigmoid or Soft step function.

		Parameters:
		-----------
		x: float
			The value used for the activation function

		Returns:
		--------
		fx: float
			The result from the activation function
		"""
		fx = 1 / (1 + np.exp(-x))
		return fx

	@property
	def function(self):
		return self._function