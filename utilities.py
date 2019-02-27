import torch

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

def accuracy(x, y):
    predictions = torch.argmax(x, dim=1)
    return (predictions == y).float().mean()

def testAccuracy(model, test_dl):
    return (sum(accuracy(model(x), y) for x, y in test_dl) / len(test_dl)).item()