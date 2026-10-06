import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    USE_BATCH_NORM = 1  # Set to 1 to use batch normalization
    DROPOUT = 0.5       # Set dropout rate between 0 and 1

class DropoutLayer(nn.Module):
    def __init__(self):
        super(DropoutLayer, self).__init__()
        self.dropout = Config.DROPOUT

    def forward(self, x):
        if self.dropout > 0 and self.training:
            out_shape = x.shape
            batch_seed_shape = out_shape[1:]  # Exclude batch dimension
            batch_seed = torch.rand(batch_seed_shape, device=x.device)
            curr_mask = (batch_seed[None, ...] > self.dropout).float()
            curr_mask = curr_mask * (1.0 / (1.0 - self.dropout))
            x = x * curr_mask
        return x

class ResidualBlock(nn.Module):
    def __init__(self, depth, conv_layer_func):
        super(ResidualBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = conv_layer_func(depth, depth)
        self.relu2 = nn.ReLU()
        self.conv2 = conv_layer_func(depth, depth)

    def forward(self, x):
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out + x

class ImpalaCNN(nn.Module):
    def __init__(self, input_shape, depths=[16, 32, 32]):
        super(ImpalaCNN, self).__init__()
        self.use_batch_norm = Config.USE_BATCH_NORM == 1
        self.depths = depths

        self.conv_sequences = nn.ModuleList()
        in_channels = input_shape[0]  # Assuming input_shape is (C, H, W)
        for depth in depths:
            seq = self._build_conv_sequence(in_channels, depth)
            self.conv_sequences.append(seq)
            in_channels = depth

        # Compute the output feature size for the fully connected layer
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Batch size 1
            out = x
            for seq in self.conv_sequences:
                out = seq(out)
            out = out.view(out.size(0), -1)
            feature_size = out.size(1)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_size, 512),
            nn.ReLU()
        )

    def _build_conv_sequence(self, in_channels, depth):
        layers = []
        layers.append(self._build_conv_layer(in_channels, depth))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(self._build_residual_block(depth))
        layers.append(self._build_residual_block(depth))
        return nn.Sequential(*layers)

    def _build_conv_layer(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(DropoutLayer())
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def _build_residual_block(self, depth):
        return ResidualBlock(depth, self._build_conv_layer)

    def forward(self, x):
        out = x
        for seq in self.conv_sequences:
            out = seq(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def create_impala_cnn():
    return ImpalaCNN(input_shape=(4, 84, 84))