import utilities
import data_visualization as dv
import quantization
import torch
import os

class ExperimentSuite:
    def __init__(self, experiments, id=""):
        self.experiments = experiments
        self.id = id

class QuantizationExperimentSuite(ExperimentSuite):
    def __init__(self, experiments, layout=dv.VerticalLayout(1250), id=""):
        super().__init__(experiments, id)
        self.visualizer = dv.Visualizer(layout, id)  # Result plots layout defined here.
        self.BACKUP_PATH = "./experiment_" + id + "_model_save.pt"
    
    def run(self):  # Note: running an experiment on a model will leave its parameters unchanged after completion. 
        for experiment in self.experiments:
            torch.save(experiment.model.state_dict(), self.BACKUP_PATH)
            quantization.quantize(experiment.model, experiment.parameter_types, experiment.quantizer_type, experiment.n, outliers_filter=experiment.outliers_filter, base=experiment.base)
            accuracy = utilities.testAccuracy(experiment.model, experiment.test_ds)
            print(experiment.id + " accuracy:", accuracy)
            self.visualizer.plot_value(accuracy, 1, "accuracy")
            for parameter_type in experiment.parameter_types:
                self.visualizer.plot_distribution(experiment.model, self.visualizer.layout.plot_resolution, experiment.id, parameter_type)
            experiment.model.load_state_dict(torch.load(self.BACKUP_PATH))
        self.visualizer.output_plots()
        os.remove(self.BACKUP_PATH)
    
class Experiment:
    def __init__(self, model, test_ds, id=""):
        self.model = model
        self.test_ds = test_ds
        self.id = id

class QuantizationExperiment(Experiment):
    def __init__(self, model, test_ds, parameter_types, quantizer_type, n, id="", outliers_filter=0, base=2):
        super().__init__(model, test_ds, id)
        self.parameter_types = parameter_types
        self.quantizer_type = quantizer_type
        self.n = n
        self.outliers_filter = outliers_filter
        self.base = base
        if self.id is "":
            if self.quantizer_type is quantization.NONE:
                self.id = "Original"
            elif self.quantizer_type is quantization.UNIFORM_A:
                self.id = "Uniform quantization (" + str(n) + ") (asymmetric)"
            elif self.quantizer_type is quantization.UNIFORM_S:
                self.id = "Uniform quantization (" + str(n) + ") (symmetric)"
            elif self.quantizer_type is quantization.LOGARITHMIC_A:
                self.id = "Logarithmic quantization (" + str(n) + ") (asymmetric)"
            elif self.quantizer_type is quantization.LOGARITHMIC_S:
                self.id = "Logarithmic quantization (" + str(n) + ") (symmetric)"
            elif self.quantizer_type is quantization.DENSITY_A:
                self.id = "Density based quantization (" + str(n) + ") (asymmetric)"
            elif self.quantizer_type is quantization.DENSITY_S:
                self.id = "Density based quantization (" + str(n) + ") (symmetric)"
            else:
                raise Exception("Unknown quantizer_type")