"""
Copyright Kaedim Limited 2021.
"""

from torch import nn


class ConvUser(nn.Module):
    """
    Common convolutional methods used by MobileCvT.

    """

    def __init__(self):
        """
        Parameters
        ----------
        None.

        """
        super().__init__()

    def conv_1x1_bn(self, inp, oup):
        """
        2D convolution and batch norm block with stride and kernel = 1.

        Parameters
        ----------
        inp : int
            Input dimension.
        oup : int
            Output dimension.

        """
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.SiLU()
        )

    def conv_nxn_bn(self, inp, oup, kernel_size=3, stride=1):
        """
        2D convolution and batch norm block with stride = n and kernel = m.

        Parameters
        ----------
        inp : int
            Input dimension.
        oup : int
            Output dimension.
        kernel_size : int
            Kernel size for conv2d.
        stride : int
            stride length.

        """
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(oup),
            nn.SiLU()
        )
