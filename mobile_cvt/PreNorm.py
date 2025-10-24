"""
Copyright Kaedim Limited 2021.
"""

from torch import nn
from .LayerNorm import LayerNorm


class PreNorm(nn.Module):
	"""
	ViT's PreNorm class.

	"""
	def __init__(self, dim, fn):
		"""
		Parameters
		----------
		dim : int
			LayerNorm dimension.
		fn : function
			Function to apply after layernorm.

		"""
		super().__init__()
		self.norm = LayerNorm(dim)
		self.fn = fn

	def forward(self, x, **kwargs):
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
		return self.fn(self.norm(x), **kwargs)
