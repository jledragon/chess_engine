"""
Copyright Kaedim Limited 2021.
"""

from torch import nn
from .Attention import Attention
from .FeedForward import FeedForward
from .PreNorm import PreNorm


class Transformer(nn.Module):
	"""
	CvT's transformer.

	"""
	def __init__(
		self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64,
		mlp_mult=4, dropout=0.):
		"""
		Parameters
		----------
		dim : int
			Attention/Feedforward input dim.
		proj_kernel : int
			padding for attention.
		kv_proj_stride : int
			Attention stride.
		depth : int
			Number of attention/feedforward layers.
		heads : int
			Number of attention heads.
		dim_head : int
			Dimension of one head.
		mlp_mult : int
			Difference between FeedForward input and hidden layer.
		dropout : float
			Dropout (default = 0.0).

		"""
		super().__init__()
		self.layers = nn.ModuleList([])
		for _ in range(depth):
			self.layers.append(nn.ModuleList([
				PreNorm(dim, Attention(
					dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
					heads=heads, dim_head=dim_head, dropout=dropout)),
				PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
			]))

	def forward(self, x):
		"""
		Performs a forward pass through the transformer.

		Parameters
		----------
		x : tensor
			Intermediate input.

		Returns
		-------
		The transformer outputs.

		"""
		for attn, ff in self.layers:
			x = attn(x) + x
			x = ff(x) + x
		return x
