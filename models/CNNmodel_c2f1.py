import torch.nn as nn
import torch

class CNNmodel_c2f1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 64, 3, 2, 1)

        self.fc1 = nn.Linear(64 * 3 * 3, 10)  # 수정된 부분

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x = self.maxpool(x) ## 절반으로 먼저

        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# model = CNNmodel_c2f1()


# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# torchsummary.summary(model, input_size=(1, 28, 28))