"""
Copyright Kaedim Limited 2021.
"""

from torch import nn


class FeedForward(nn.Module):
	"""
	ViT's FeedForward class.

	"""
	def __init__(self, dim, mult=4, dropout=0.):
		"""
		Parameters
		----------
		dim : int
			Input dimension.
		mult : int
			Hidden dimension ratio.
		dropout : float
			dropout parameter (default = 0.0).

		"""
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(dim, dim * mult, 1),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Conv2d(dim * mult, dim, 1),
			nn.Dropout(dropout)
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
