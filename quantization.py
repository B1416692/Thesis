import utilities

class Quantizer:
    def __init__(self, model, parameter_type):
        self.model = model
        self.parameter_type = parameter_type

    def quantize(self, element):
        raise NotImplementedError("I'm still working things out...")

class DomainQuantizer(Quantizer):
    def __init__(self, model, parameter_type, n):
        super().__init__(model, parameter_type)
        self.n = n
        self.quantizationDomain = [0]

    def quantize(self, element):
        return min(self.quantizationDomain, key=lambda x:abs(x-(round(element, 1))))

# TODO: For now, values is assumed to contain elments BOTH greater and smaller than 0. Make it universal.
# TODO: Also, values is supposed to be non empthy. Change.
class AsymmetricLinearQuantizer(DomainQuantizer):
    def __init__(self, model, parameter_type, n):
        super().__init__(model, parameter_type, n)
        values = utilities.get(self.model, self.parameter_type)
        min_value = min(values)
        max_value = max(values)
        negative_gap = min_value / ((self.n - 1) / 2)
        positive_gap = max_value / ((self.n - 1) / 2)
        for i in range(1, int((self.n - 1) / 2) + 1):
            self.quantizationDomain.append(i * negative_gap)
            self.quantizationDomain.append(i * positive_gap)