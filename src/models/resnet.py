import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, channels_in, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or channels_in != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channel_factor=1, dropout=0.0):
        super(ResNet, self).__init__()
        self.channels_in = int(64 * channel_factor)
        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, int(64 * channel_factor), kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64 * channel_factor))
        self.layer1 = self._make_layer(block, int(64 * channel_factor), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * channel_factor), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(265 * channel_factor), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * channel_factor), num_blocks[3], stride=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(512 * channel_factor), num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.channels_in, channels, stride))
            self.channels_in = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        out = nn.functional.dropout(out, p=self.dropout, training=self.training)
        out = self.linear(out)
        out = nn.functional.softmax(out, dim=1)
        return out


def resnet_custom(num_layers, output_dim=10, channel_factor=1, dropout=0.0, **kwargs):
    """
    Create a ResNet model with a specified number of layers.

    Parameters:
        num_layers (int): Number of layers. Should be one of [18, 34, 50, 101, 152].
        output_dim (int): Number of output classes for classification.
        channel_factor (int): Factor to scale the number of channels in the network.

    Returns:
        model (nn.Module): The ResNet model.
    """
    if num_layers == 10:
        return ResNet(BasicBlock, [1, 1, 1, 1], output_dim, channel_factor, dropout=dropout)
    if num_layers == 18:
        return ResNet(BasicBlock, [2, 2, 2, 2], output_dim, channel_factor, dropout=dropout)
    elif num_layers == 34:
        return ResNet(BasicBlock, [3, 4, 6, 3], output_dim, channel_factor, dropout=dropout)
    elif num_layers == 50:
        return ResNet(Bottleneck, [3, 4, 6, 3], output_dim)
    elif num_layers == 101:
        return ResNet(Bottleneck, [3, 4, 23, 3], output_dim)
    elif num_layers == 152:
        return ResNet(Bottleneck, [3, 8, 36, 3], output_dim)
    else:
        raise ValueError("Unsupported ResNet model depth. Choose from [18, 34, 50, 101, 152].")


class Bottleneck(nn.Module):
    def __init__(self, channels_in, channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or channels_in != channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


# Example usage:
# Create a ResNet with 18 layers
#model = resnet_custom(num_layers=10, output_dim=10, channel_factor=0.5)
#print(model)
