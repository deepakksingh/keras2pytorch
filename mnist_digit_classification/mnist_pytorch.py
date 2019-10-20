
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import torch

batch_size = 128
num_epochs = 12

# Note that Torch has channels first by default (M, C, W, H) whereas Keras has
# channels last (M, W, H, C)
class mnist_cnn(nn.Module):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        # input shape = (1,28,28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # output shape = (32,26,26) where 26 = (28-3+1)/1
        # Params = ((3*3)*1 +  1)*32 = 320
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # output shape = (64,24,24) where 24 = (26-3+1)/1
        # Params = ((3*3)*32 + 1)*64 = 18496
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # output shape = (64,12,12), Params = 0
        self.dropout1 = nn.Dropout(p=0.25)
        # output shape = (64,12,12), Params = 0
        self.flatten = nn.Flatten()
        # output shape = (9216,), Params = 0
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        # output shape = (128,)
        # Params = (9216+1)*128 = 1179776
        self.dropout2 = nn.Dropout(p=0.5)
        # output shape = (128,), Params = 0
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # output shape = (10,)
        # Params = (128+1)*10 = 1290

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
