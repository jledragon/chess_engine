"""
Copyright Kaedim Limited 2021.
"""

from einops.layers.torch import Reduce
from .ConvUser import ConvUser

from torch import nn
from .MV2Block import MV2Block
from.MobileCvTBlock import MobileCvTBlock


class MobileCvT(ConvUser):
    """
    MobileCvT.

    Paper (MobileViT): https://arxiv.org/abs/2110.02178
    Paper (CvT): https://arxiv.org/abs/2103.15808
    Based on (MobileViT): https://github.com/chinhsuanwu/mobilevit-pytorch

    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3),
        dataset_dim=3,
        dropout=0.1
    ):
        """
        Parameters
        ----------
        image_size : (int, int)
            The image width and height.
        dims : int[]
            MobileCvT dimensions.
        channels : int[]
            MV2Block channels.
        num_classes : int
            Output c_dim.
        expansion : int
            Expansion for MV2Block.
        kernel_size : int
            The conv kernel size for MV2Block.
        patch_size : (int, int):
            MobileCvTBlock patch size.
        depths : (int, int, int):
            MobileCvTBlock depths.
        dataset_dim : int
            Number of channels of input image.

        """
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = self.conv_nxn_bn(dataset_dim, init_dim, stride=1)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        # Local board view
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 1, expansion),
            MobileCvTBlock(
                dims[0], depths[0], channels[5],
                kernel_size=3, patch_size=patch_size, dropout=dropout)
        ]))

        # bigger board view
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileCvTBlock(
                dims[1], depths[1], channels[7],
                kernel_size=3, patch_size=patch_size, dropout=dropout)
        ]))

        # near-global board view
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileCvTBlock(
                dims[2], depths[2], channels[9],
                kernel_size=3, patch_size=patch_size, dropout=dropout)
        ]))
        
        # consolidate to large area
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[9], channels[10], 1, expansion),
            MobileCvTBlock(
                dims[2], depths[2], channels[11],
                kernel_size=3, patch_size=patch_size, dropout=dropout)
        ]))

        # consolidate to local
        self.to_logits = nn.Sequential(
            self.conv_1x1_bn(channels[-2], last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Parameters
        ----------
        x : tensor
            Image input.

        Returns
        -------
        The neural network outputs.

        """
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)
