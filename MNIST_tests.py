# - Download MNIST

from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

# - Load dataset

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# - Setup dataset

import torch
from torch.utils.data import TensorDataset

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

# - Utilities
from torch.utils.data import DataLoader

def get_data(train_ds, valid_ds, batch_size):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(valid_ds, batch_size=batch_size * 2),
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get(model, parameter_type):
    result = []
    for name, param in model.named_parameters():
        if parameter_type in name:
            parameters = torch.flatten(param)
            for parameter in parameters:
                parameter = parameter.item()
                result.append(parameter)
    return result

def fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs):
    print("Training...")
    print("#", "\t", "Loss")
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            prediction = model(x)
            loss = loss_func(prediction, y)

            loss.backward()
            opt.step()
            opt.zero_grad()
        
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(x), y) for x, y in valid_dl)

        print(epoch + 1, "\t", (valid_loss / len(valid_dl)).item())

def accuracy(x, y):
    predictions = torch.argmax(x, dim=1)
    return (predictions == y).float().mean()

# - Data visualization

import numpy as np
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

DISITRBUTION_PLOT_WIDTH = 1250
distribution_plots = {}

def make_plot(title, hist, edges, x, x_label, y_label):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="steelblue", line_color="steelblue", alpha=1.0)

    p.y_range.start = 0
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.grid.grid_line_color="white"
    return p

def plot_distribution(model, resolution, title, parameter_type):
    values = get(model, parameter_type)
    minimum = min(values)
    maximum = max(values)
    
    hist, edges = np.histogram(values, density=True, bins=resolution)
    x = np.linspace(minimum, maximum, resolution)

    plot = make_plot(title + " - " + parameter_type, hist, edges, x, parameter_type, "frequency")
    if parameter_type not in distribution_plots.keys():
        distribution_plots[parameter_type] = []
    distribution_plots[parameter_type].append(plot)

# - Define neural network structures

from torch import nn
import math
from kafnets import KAF, KAF2D

class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 15)
        self.linear4 = nn.Linear(15, 15)
        self.linear5 = nn.Linear(15, 10)

    def forward(self, x):
        y1 = F.relu(self.linear1(x))
        y2 = F.relu(self.linear2(y1))
        y3 = F.relu(self.linear3(y2))
        y4 = F.relu(self.linear4(y3))
        y = F.log_softmax(self.linear5(y4), dim=0)
        return y

class FF_KAF(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 20)
        self.kaf1 = KAF(20)
        self.linear2 = nn.Linear(20, 15)
        self.kaf2 = KAF(15)
        self.linear3 = nn.Linear(15, 10)

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.kaf1(y1)
        y3 = self.linear2(y2)
        y4 = self.kaf2(y3)
        y = F.log_softmax(self.linear3(y4), dim=0)
        return y

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(12, 10, kernel_size=5, stride=2, padding=2)
        self.linear1 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        y1 = F.relu(self.conv1(x))
        y2 = F.relu(self.conv2(y1))
        y3 = F.relu(self.conv3(y2))
        y4 = F.avg_pool2d(y3, 4)
        y4 = y4.view(-1, y4.size(1))
        y = F.log_softmax(self.linear1(y4))
        return y

class CNN_KAF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(12, 10, kernel_size=5, stride=2, padding=2)
        self.kaf1 = KAF(10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        y1 = F.relu(self.conv1(x))
        y2 = F.relu(self.conv2(y1))
        y3 = F.relu(self.conv3(y2))
        y4 = F.avg_pool2d(y3, 4)
        y4 = y4.view(-1, y4.size(1))
        y = F.log_softmax(self.kaf1(y4), dim=0)
        return y

# - Configure tests

from torch import optim
import torch.nn.functional as F

# FF
model = FF()  # Model.
lr = 0.5  # Learning rate.
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.2)  # Optimizer.
loss_func = F.nll_loss  # Loss function.
# Fixed parameters
batch_size = 64  # Batch size.
train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
epochs = 5  # How many epochs to train for.

print("Model:", model)
print("Number of parameters:", count_parameters(model))
fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs)
print("Accuracy:", (sum(accuracy(model(x), y) for x, y in valid_dl) / len(valid_dl)).item())
plot_distribution(model, DISITRBUTION_PLOT_WIDTH, "FF", "weight")
plot_distribution(model, DISITRBUTION_PLOT_WIDTH, "FF", "bias")
print("")

# FF_KAF
model = FF_KAF()  # Model.
lr = 0.1  # Learning rate.
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.1)  # Optimizer.

print("Model:", model)
print("Number of parameters:", count_parameters(model))
fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs)
print("Accuracy:", (sum(accuracy(model(x), y) for x, y in valid_dl) / len(valid_dl)).item())
plot_distribution(model, DISITRBUTION_PLOT_WIDTH, "FF_KAF", "weight")
plot_distribution(model, DISITRBUTION_PLOT_WIDTH, "FF_KAF", "bias")
print("")

# CNN
model = CNN()  # Model.
lr = 0.1  # Learning rate.
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # Optimizer.

print("Model:", model)
print("Number of parameters:", count_parameters(model))
fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs)
print("Accuracy:", (sum(accuracy(model(x), y) for x, y in valid_dl) / len(valid_dl)).item())
plot_distribution(model, DISITRBUTION_PLOT_WIDTH, "CNN", "weight")
plot_distribution(model, DISITRBUTION_PLOT_WIDTH, "CNN", "bias")
print("")

# CNN_KAF
model = CNN_KAF()
lr = 0.1  # Learning rate.
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # Optimizer.

print("Model:", model)
print("Number of parameters:", count_parameters(model))
fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs)
print("Accuracy:", (sum(accuracy(model(x), y) for x, y in valid_dl) / len(valid_dl)).item())
plot_distribution(model, DISITRBUTION_PLOT_WIDTH, "CNN_KAF", "weight")
plot_distribution(model, DISITRBUTION_PLOT_WIDTH, "CNN_KAF", "bias")
print("")

# Output plots
for key in distribution_plots.keys():
    output_file(key + '_distribution.html', title=key + " distribution")
    show(gridplot(distribution_plots[key], ncols=1, plot_width=DISITRBUTION_PLOT_WIDTH, plot_height=400, toolbar_location=None))