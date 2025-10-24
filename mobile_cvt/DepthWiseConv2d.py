"""
Copyright Kaedim Limited 2021.
"""

from torch import nn


class DepthWiseConv2d(nn.Module):
	"""
	CvT's DepthWiseConv2d class.

	"""
	def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
		"""
		Parameters
		----------
		dim_in : int
			Input dimension.
		dim_out : int
			Output dimension.
		kernel_size : int
			Convolutional kernel size.
		padding : int
			Convolutional padding.
		stride : int
			Convolutional stride.
		bias : bool
			Whether to use bias in conv2d.

		"""
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(
				dim_in, dim_in, kernel_size=kernel_size, padding=padding,
				groups=dim_in, stride=stride, bias=bias),
			nn.BatchNorm2d(dim_in),
			nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
		)

	def forward(self, x):
		"""
		Performs a forward pass through the model.

		Parameters
		----------
		x : tensor
			Intermediate input.

		Returns
		-------
		The neural network outputs.

		"""
		return self.net(x)
