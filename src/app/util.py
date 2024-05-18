import torch
from torch import nn
import torch.nn.functional as F

image_size = 224
num_classes = 5

def conv2d_output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1

def maxpool2d_output_size(input_size, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size
    return (input_size - kernel_size + 2 * padding) // stride + 1

def get_flatten_size(input_size, layers):
    size = input_size
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            size = conv2d_output_size(size, layer.kernel_size[0], stride=layer.stride[0], padding=layer.padding[0])
        elif isinstance(layer, nn.MaxPool2d):
            size = maxpool2d_output_size(size, layer.kernel_size, stride=layer.stride, padding=layer.padding)
    return size

class CNNModel1(nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        flatten_size = get_flatten_size(image_size, [self.conv1, self.pool1, self.conv2, self.pool2])
        self.fc1 = nn.Linear(32 * flatten_size**2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x