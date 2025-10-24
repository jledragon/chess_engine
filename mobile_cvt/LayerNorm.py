# -*- coding: utf-8 -*-
"""
Created on Sun Mar  10 22:40:32 2024

@author: jledragon
"""

from torch import nn
import torch


class LayerNorm(nn.Module):
	"""
	layernorm, but done in the channel dimension #1
	Used by CvT to slightly modify the behaviour of base LayerNorm.

	"""
	def __init__(self, dim, eps=1e-5):
		"""
		Parameters
		----------
		dim : int
			Input dimension.
		eps : float, optional
			Stabilisation parameter. The default is 1e-5.

		"""
		super().__init__()
		self.eps = eps
		self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
		self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

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
		var = torch.var(x, dim=1, unbiased=False, keepdim=True)
		mean = torch.mean(x, dim=1, keepdim=True)
		return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
