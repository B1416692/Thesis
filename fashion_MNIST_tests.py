# - Dataset

import torchvision

train_ds = torchvision.datasets.FashionMNIST(root="./data/FashionMNIST", train=True, transform=torchvision.transforms.ToTensor(), download=True)
valid_ds = torchvision.datasets.FashionMNIST(root="./data/FashionMNIST", train=False, transform=torchvision.transforms.ToTensor())

from torch.utils.data import DataLoader

def get_data(train_ds, valid_ds, batch_size):
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(valid_ds, batch_size=batch_size * 2))

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
        x = x.view(-1, 28 * 28).requires_grad_()
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
        x = x.view(-1, 28 * 28).requires_grad_()
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

# - Train function

import torch

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

# - Experiments
# Notice that models parameters will remain unchanged after an experiment, to allow easy experiments serialization.
# To change the parameters in a persistant way use modification functions directly on the model, for example a quantization function.

from torch import optim
import torch.nn.functional as F
import utilities
import experiment_suite
import quantization
import data_visualization as dv

LOAD_MODELS = False  # If False, models parameters will be saved after training. If True, models parameters will be loaded.

# Fixed parameters
batch_size = 64  # Batch size.
train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
epochs = 5  # How many epochs to train for.
loss_func = F.nll_loss  # Loss function.

LAYOUT_WIDTH = 1250  # Width of results plots.

# TODO: Since torchvision.dataset.MNIST images get actually stored in 28 by 28 tensors instead of 728 tensors, FF networks must be refactored.

# FF
model = FF()  # Model.
lr = 0.5  # Learning rate.
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.2)  # Optimizer.

print("Model:", model)
print("Number of parameters:", utilities.count_parameters(model))
model_name = "FashionMNIST_FF"
if LOAD_MODELS is False:
    print("Accuracy before training:", utilities.testAccuracy(model, valid_dl))
    fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs)
    torch.save(model.state_dict(), "./" + model_name + ".pt")
    print("Accuracy after training:", utilities.testAccuracy(model, valid_dl))
else:
    model.load_state_dict(torch.load("./" + model_name +".pt"))
    print("Parameters loaded")

experiments = []
parameters_to_quantize = ["weight"]
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.NONE, 0))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 31))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 17))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 9))
suite = experiment_suite.QuantizationExperimentSuite(experiments, layout=dv.SplitLayoutPlus(LAYOUT_WIDTH, len(parameters_to_quantize)), id="FashionMNIST FF")
suite.run()

# FF_KAF
model = FF_KAF()  # Model.
lr = 0.1  # Learning rate.
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.1)  # Optimizer.

print("Model:", model)
print("Number of parameters:", utilities.count_parameters(model))
model_name = "FashionMNIST_FF_KAF"
if LOAD_MODELS is False:
    print("Accuracy before training:", utilities.testAccuracy(model, valid_dl))
    fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs)
    torch.save(model.state_dict(), "./" + model_name + ".pt")
    print("Accuracy after training:", utilities.testAccuracy(model, valid_dl))
else:
    model.load_state_dict(torch.load("./" + model_name +".pt"))
    print("Parameters loaded")

experiments = []
parameters_to_quantize = ["weight", "alpha"]
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.NONE, 0))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 31))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 17))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 9))
suite = experiment_suite.QuantizationExperimentSuite(experiments, layout=dv.SplitLayoutPlus(LAYOUT_WIDTH, len(parameters_to_quantize)), id="FashionMNIST FF_KAF")
suite.run()

# CNN
model = CNN()  # Model.
lr = 0.1  # Learning rate.
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # Optimizer.

print("Model:", model)
print("Number of parameters:", utilities.count_parameters(model))
model_name = "FashionMNIST_CNN"
if LOAD_MODELS is False:
    print("Accuracy before training:", utilities.testAccuracy(model, valid_dl))
    fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs)
    torch.save(model.state_dict(), "./" + model_name + ".pt")
    print("Accuracy after training:", utilities.testAccuracy(model, valid_dl))
else:
    model.load_state_dict(torch.load("./" + model_name +".pt"))
    print("Parameters loaded")

experiments = []
parameters_to_quantize = ["weight"]
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.NONE, 0))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 31))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 17))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 9))
suite = experiment_suite.QuantizationExperimentSuite(experiments, layout=dv.SplitLayoutPlus(LAYOUT_WIDTH, len(parameters_to_quantize)), id="FashionMNIST CNN")
suite.run()

# CNN_KAF
model = CNN_KAF()
lr = 0.1  # Learning rate.
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # Optimizer.

print("Model:", model)
print("Number of parameters:", utilities.count_parameters(model))
model_name = "FashionMNIST_CNN_KAF"
if LOAD_MODELS is False:
    print("Accuracy before training:", utilities.testAccuracy(model, valid_dl))
    fit(model, lr, opt, loss_func, batch_size, train_dl, valid_dl, epochs)
    torch.save(model.state_dict(), "./" + model_name + ".pt")
    print("Accuracy after training:", utilities.testAccuracy(model, valid_dl))
else:
    model.load_state_dict(torch.load("./" + model_name +".pt"))
    print("Parameters loaded")

experiments = []
parameters_to_quantize = ["weight", "alpha"]
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.NONE, 0))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 31))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 17))
experiments.append(experiment_suite.QuantizationExperiment(model, valid_dl, parameters_to_quantize, quantization.UNIFORM_A, 9))
suite = experiment_suite.QuantizationExperimentSuite(experiments, layout=dv.SplitLayoutPlus(LAYOUT_WIDTH, len(parameters_to_quantize)), id="FashionMNIST CNN_KAF")
suite.run()