"""Custom CNN for multi-label food recognition (one sigmoid output per class)."""
import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Two 3x3 convs with a skip connection; downsamples when stride > 1."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(cin, cout, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class FoodCNN(nn.Module):
    """Small ResNet-style CNN trained from scratch.

    Input:  (B, 3, H, W) normalized image (224x224 by default, any size works
            thanks to global pooling).
    Output: (B, num_classes) raw logits. Apply sigmoid for per-class probabilities;
            do NOT softmax - one fridge photo contains many items at once.
    """

    def __init__(self, num_classes, widths=(32, 64, 128, 256, 384), dropout=0.3):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(ConvBlock(3, w[0], stride=2), ConvBlock(w[0], w[0]))
        self.stages = nn.Sequential(
            ResBlock(w[0], w[1], 2), ResBlock(w[1], w[1]),
            ResBlock(w[1], w[2], 2), ResBlock(w[2], w[2]),
            ResBlock(w[2], w[3], 2), ResBlock(w[3], w[3]),
            ResBlock(w[3], w[4], 2), ResBlock(w[4], w[4]),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(w[4], num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.pool(self.stages(self.stem(x))))


if __name__ == "__main__":
    net = FoodCNN(57)
    out = net(torch.randn(2, 3, 224, 224))
    print(out.shape, f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")
