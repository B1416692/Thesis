import utilities
import numpy as np

NONE = -1
UNIFORM_A = 0
UNIFORM_S = 1
LOGARITHMIC_A = 2
LOGARITHMIC_S = 3
DENSITY_A = 4
DENSITY_S = 5

def quantize(model, parameter_types, quantizer_type, n, outliers_filter=0, base=2):
    quantizers = {}
    for parameter_type in parameter_types:
        quantizer = None
        if quantizer_type is NONE:
            quantizer = IdentityQuantizer(model, parameter_type)
        elif quantizer_type is UNIFORM_A:
            quantizer = AsymmetricUniformQuantizer(model, parameter_type, n)
        elif quantizer_type is UNIFORM_S:
            quantizer = SymmetricUniformQuantizer(model, parameter_type, n)
        elif quantizer_type is LOGARITHMIC_A:
            quantizer = AsymmetricLogarithmicQuantizer(model, parameter_type, n, base=base)
        elif quantizer_type is LOGARITHMIC_S:
            quantizer = SymmetricLogarithmicQuantizer(model, parameter_type, n, base=base)
        elif quantizer_type is DENSITY_A:
            quantizer = AsymmetricDensityBasedQuantizer(model, parameter_type, n)
        elif quantizer_type is DENSITY_S:
            quantizer = SymmetricDensityBasedQuantizer(model, parameter_type, n)
        else:
            raise Exception("Unknown quantizer_type")
        quantizers[parameter_type] = quantizer
    if "weight" in parameter_types:
        quantizer = quantizers["weight"]
        for layer in model.children():
            if hasattr(layer, "weight"):
                layer.weight.data.apply_(quantizer.quantize)  # apply_(function) only works with CPU tensors.
    # TODO: Phugly. Find way to avoid these repetitions.
    if "alpha" in parameter_types:
        quantizer = quantizers["alpha"]
        for layer in model.children():
            if hasattr(layer, "alpha"):
                layer.alpha.data.apply_(quantizer.quantize)  # apply_(function) only works with CPU tensors.

class Quantizer:
    def __init__(self, model, parameter_type):
        self.model = model
        self.parameter_type = parameter_type

    def quantize(self, element):
        raise NotImplementedError("I'm still working things out...")

# Identity quantizer. Actually applies no quantization. Utility.
class IdentityQuantizer(Quantizer):
    def __init__(self, model, parameter_type):
        super().__init__(model, parameter_type)

    def quantize(self, element):
        return element


class RangeQuantizer(Quantizer):
    def __init__(self, model, parameter_type, n):
        super().__init__(model, parameter_type)
        self.n = n
        self.elements_per_side = int((self.n - 1) / 2)
        self.quantizationDomain = [0]

    def quantize(self, element):
        return min(self.quantizationDomain, key=lambda x:abs(x-element))

# TODO: For now, values is assumed to contain elments BOTH greater and smaller than 0. Make it universal.
# TODO: Also, values is supposed to be non empthy. Check.
class AsymmetricUniformQuantizer(RangeQuantizer):
    def __init__(self, model, parameter_type, n):
        super().__init__(model, parameter_type, n)
        values = utilities.get(self.model, self.parameter_type)
        min_value = min(values)
        max_value = max(values)
        negative_gap = min_value / self.elements_per_side
        positive_gap = max_value / self.elements_per_side
        for i in range(1, self.elements_per_side + 1):
            self.quantizationDomain.append(i * negative_gap)
            self.quantizationDomain.append(i * positive_gap)

class SymmetricUniformQuantizer(RangeQuantizer):
    def __init__(self, model, parameter_type, n):
        super().__init__(model, parameter_type, n)
        values = utilities.get(self.model, self.parameter_type)
        min_value = min(values)
        max_value = max(values)
        maximum = max(abs(min_value), abs(max_value))
        gap = maximum / self.elements_per_side
        for i in range(1, self.elements_per_side + 1):
            self.quantizationDomain.append(i * gap)
            self.quantizationDomain.append(-i * gap)

class AsymmetricLogarithmicQuantizer(RangeQuantizer):
    def __init__(self, model, parameter_type, n, base=2):
        super().__init__(model, parameter_type, n)
        values = utilities.get(self.model, self.parameter_type)
        min_value = min(values)
        max_value = max(values)
        total = 0
        for i in range(0, self.elements_per_side):
            total += (base ** i)
        negative_scale = abs(min_value) / total
        positive_scale = abs(max_value) / total
        positive_total = 0
        negative_total = 0
        for i in range(0, self.elements_per_side):
            negative_total += (base ** i)
            self.quantizationDomain.append(-negative_total * negative_scale)
            positive_total += (base ** i)
            self.quantizationDomain.append(positive_total * positive_scale)
        self.quantizationDomain.sort()

class SymmetricLogarithmicQuantizer(RangeQuantizer):
    def __init__(self, model, parameter_type, n, base=2):
        super().__init__(model, parameter_type, n)
        values = utilities.get(self.model, self.parameter_type)
        min_value = min(values)
        max_value = max(values)
        maximum = max(abs(min_value), abs(max_value))
        total = 0
        for i in range(0, self.elements_per_side):
            total += (base ** i)
        scale = maximum / total
        total = 0
        for i in range(0, self.elements_per_side):
            total += (base ** i)
            self.quantizationDomain.append(-total * scale)
            self.quantizationDomain.append(total * scale)
        self.quantizationDomain.sort()

import bisect

# Note: it will probably has not 0.
# TODO: maybe forcing the two smallest (positive and negative) values to be equal, we get back 0 and improve accuracy?
class AsymmetricDensityBasedQuantizer(RangeQuantizer):
    def __init__(self, model, parameter_type, n):
        super().__init__(model, parameter_type, n + 2)
        self.quantizationDomain = []  # Remove 0 as near 0 will be obtained by averaging the two smaller (absolute) values.
        values = utilities.get(self.model, self.parameter_type)
        negative_values = list(filter(lambda x: x <=0, values))
        negative_values = [abs(value) for value in negative_values]
        positive_values = list(filter(lambda x: x >=0, values))
        percentile_gap = 100 / self.elements_per_side
        for i in range(1, self.elements_per_side):
            self.quantizationDomain.append(-np.percentile(negative_values, 100 - percentile_gap * i))
            self.quantizationDomain.append(np.percentile(positive_values, 100 - percentile_gap * i))
        self.quantizationDomain.append(-max(negative_values) - 0.00001)
        self.quantizationDomain.append(max(positive_values) + 0.00001)
        self.quantizationDomain.sort()
    
    def quantize(self, element):
        index = bisect.bisect(self.quantizationDomain, element)
        return (self.quantizationDomain[index-1] + self.quantizationDomain[index]) / 2

# Note: there is 0 but the distribution is not uniform.
class SymmetricDensityBasedQuantizer(RangeQuantizer):
    def __init__(self, model, parameter_type, n):
        super().__init__(model, parameter_type, n + 2)
        self.quantizationDomain = []  # Remove 0 as it will be obtained by averaging the two smaller (absolute) values.
        values = utilities.get(self.model, self.parameter_type)
        negative_values = list(filter(lambda x: x <=0, values))
        negative_values = [abs(value) for value in negative_values]
        positive_values = list(filter(lambda x: x >=0, values))
        maximum_negative = max(negative_values)
        maximum_positive = max(positive_values)
        to_use_values = []
        if maximum_negative > maximum_positive:
            to_use_values = negative_values
        else:
            to_use_values = positive_values
        percentile_gap = 100 / self.elements_per_side
        for i in range(1, self.elements_per_side):
            self.quantizationDomain.append(-np.percentile(to_use_values, 100 - percentile_gap * i))
            self.quantizationDomain.append(np.percentile(to_use_values, 100 - percentile_gap * i))
        self.quantizationDomain.append(-max(negative_values) - 0.00001)
        self.quantizationDomain.append(max(positive_values) + 0.00001)
        self.quantizationDomain.sort()
    
    def quantize(self, element):
        index = bisect.bisect(self.quantizationDomain, element)
        return (self.quantizationDomain[index-1] + self.quantizationDomain[index]) / 2