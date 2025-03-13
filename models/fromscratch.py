import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = self.conv_block(in_channel, 64)
        self.conv2 = self.conv_block(64, 32)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
                nn.Linear(32 * 3 * 3, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(self.conv2(x))
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x

    def conv_block(self, in_chan, out_chan):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            )