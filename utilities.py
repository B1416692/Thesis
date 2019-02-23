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