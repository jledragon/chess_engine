# -*- coding: utf-8 -*-
"""
Created on Sun Mar  10 22:40:32 2024

@author: jledragon
"""

import torch
from .Transformer import Transformer
from .LayerNorm import LayerNorm
from .ConvUser import ConvUser


class MobileCvTBlock(ConvUser):
    """
    Mobile CvT block, based on Mobile ViT block, but using CvT's transformer.

    """
    def __init__(self, dim, depth, channel, kernel_size, patch_size, dropout=0.):
        """
        Parameters
        ----------
        dim : int
            Transformer dimension.
        depth : int
            Transformer depth.
        channel : int
            Convolutional hidden dimension.
        kernel_size : int
            Size of the sliding window in conv1.
        patch_size : int
            Vision transformer patch size.
        dropout : float
            dropout. Default = 0.0

        """
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = self.conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = self.conv_1x1_bn(channel, dim)

        self.transformer = Transformer(
            dim=dim, proj_kernel=3, kv_proj_stride=2, depth=depth, heads=4,
            mlp_mult=4, dropout=dropout)

        self.conv3 = self.conv_1x1_bn(dim, channel)
        self.conv4 = self.conv_nxn_bn(2 * channel, channel, kernel_size)
        self.ln = LayerNorm(dim)

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
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        x = self.ln(x)
        x = self.transformer(x)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x
