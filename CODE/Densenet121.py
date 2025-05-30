import torch
# torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F

class _DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm3d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout2 = nn.Dropout3d(0.1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.dropout2(self.conv2(self.relu2(self.norm2(out))))
        return torch.cat([x, out], 1)


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer3D(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet3D(nn.Module):
    """
    A minimal 3D DenseNet-121 style architecture for demonstration.
    """
    def __init__(self, init_channels=64, growth_rate=32, block_layers=(6, 12, 24, 16), out_dim=256):
        """
        - block_layers corresponds to the number of layers in each of the 4 dense blocks.
        - out_dim is the final embedding dimension for each patch.
        """
        super().__init__()
        self.init_channels = init_channels
        # Stem
        self.conv1 = nn.Conv3d(1, init_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.norm1 = nn.BatchNorm3d(init_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Dense Blocks
        num_features = init_channels
        self.denseblock1 = _DenseBlock3D(block_layers[0], num_features, growth_rate)
        num_features += block_layers[0] * growth_rate
        self.transition1 = _Transition3D(num_features, num_features // 2)
        num_features = num_features // 2

        self.denseblock2 = _DenseBlock3D(block_layers[1], num_features, growth_rate)
        num_features += block_layers[1] * growth_rate
        self.transition2 = _Transition3D(num_features, num_features // 2)
        num_features = num_features // 2

        # self.denseblock3 = _DenseBlock3D(block_layers[2], num_features, growth_rate)
        # num_features += block_layers[2] * growth_rate
        # self.transition3 = _Transition3D(num_features, num_features // 2)
        # num_features = num_features // 2

        self.denseblock4 = _DenseBlock3D(block_layers[3], num_features, growth_rate)
        num_features += block_layers[3] * growth_rate

        # Final batch norm
        self.norm_final = nn.BatchNorm3d(num_features)

        # Global average pool & Linear
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(num_features, out_dim)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.pool1(self.relu(self.norm1(x)))

        # Dense Blocks
        x = self.transition1(self.denseblock1(x))
        x = self.transition2(self.denseblock2(x))
        # x = self.transition3(self.denseblock3(x))
        x = self.denseblock4(x)

        x = self.relu(self.norm_final(x))
        x = self.avgpool(x)         # (B, num_features, 1, 1, 1)
        x = torch.flatten(x, 1)     # (B, num_features)
        out = self.fc(x)            # (B, out_dim)
        return out