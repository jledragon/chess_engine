# -*- coding: utf-8 -*-
"""
Created on Sun Mar  10 22:40:32 2024

@author: jledragon
"""

from torch import nn


class MV2Block(nn.Module):
	"""
	MV2 block described in MobileNetV2.
	Paper: https://arxiv.org/pdf/1801.04381
	Based on: https://github.com/tonylins/pytorch-mobilenet-v2

	"""

	def __init__(self, inp, oup, stride=1, expansion=4):
		"""
		Parameters
		----------
		inp : int
			Input dimension.
		out : int
			Output dimension.
		stride : int
			Convolutional stride.
		expansion : int
			The difference between input and hidden dimensions.

		"""
		super().__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = int(inp * expansion)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expansion == 1:
			self.conv = nn.Sequential(
				# dw
				nn.Conv2d(
					hidden_dim, hidden_dim, 3, stride,
					1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.SiLU(),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
				# pw
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.SiLU(),
				# dw
				nn.Conv2d(
					hidden_dim, hidden_dim, 3, stride,
					1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.SiLU(),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		"""
		Performs a forward pass through the block.

		Parameters
		----------
		x : tensor
			Intermediate input.

		Returns
		-------
		The block outputs.

		"""
		out = self.conv(x)
		if self.use_res_connect:
			out = out + x
		return out
