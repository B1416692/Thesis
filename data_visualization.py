import numpy as np
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
import utilities
import math

class Visualizer:
    def __init__(self, layout, id):
        self.layout = layout
        self.id = id
        self.distribution_plots = {}

    # - "Physically" makes a graph.
    def make_plot(self, title, hist, edges, x, x_label, y_label, color, ghost_hist=None):
        p = figure(title=title, tools='', background_fill_color="#fafafa")

        # If ghost_hist is present, plot it behind the original, with dimmed color.
        if ghost_hist is not None:
            p.quad(top=ghost_hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=color, line_color=color, alpha=0.15)

        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=color, line_color=color, alpha=1.0)

        p.y_range.start = 0
        p.xaxis.axis_label = x_label
        p.yaxis.axis_label = y_label
        p.grid.grid_line_color="white"
        return p

    # - Makes a distribution graph.
    def plot_distribution(self, model, resolution, title, parameter_type, ghost_model=None):
        values = utilities.get(model, parameter_type)
        minimum = min(values)
        maximum = max(values)

        ghost_hist = None
        # If ghost_model is present, plot it along with the original.
        if ghost_model is not None:
            original_values = utilities.get(ghost_model, parameter_type)
            original_minimum = min(original_values)
            original_maximum = max(original_values)
            minimum = min(minimum, original_minimum)
            maximum = min(maximum, original_maximum)
            ghost_hist, edges2 = np.histogram(original_values, density=True, bins=resolution)

        
        hist, edges = np.histogram(values, density=True, bins=resolution)
        
        x = np.linspace(minimum, maximum, resolution)

        color = "orangered"
        if parameter_type is "weight":
            color = "steelblue"
        elif parameter_type is "alpha":
            color = "darkorange"
        elif parameter_type is "bias":
            color = "silver"

        plot = self.make_plot(title + " - " + parameter_type, hist, edges, x, parameter_type, "frequency", color, ghost_hist=ghost_hist)
        if parameter_type not in self.distribution_plots.keys():
            self.distribution_plots[parameter_type] = []
        self.distribution_plots[parameter_type].append(plot)

    def output_plots(self):
        if self.layout.__class__ is VerticalLayout:
            output_file(self.id + ".html", title=self.id)
            distribution_plots = []
            for key in self.distribution_plots.keys():
                distribution_plots += self.distribution_plots[key]
            show(gridplot(distribution_plots, ncols=self.layout.ncols, plot_width=self.layout.plot_width, plot_height=self.layout.plot_eight, toolbar_location=None))
        elif self.layout.__class__ is SeparateLayout:
            for key in self.distribution_plots.keys():
                output_file(self.id + "_" + key +".html", title=self.id + " - " + key)
                show(gridplot(self.distribution_plots[key], ncols=self.layout.ncols, plot_width=self.layout.plot_width, plot_height=self.layout.plot_eight, toolbar_location=None))
        elif self.layout.__class__ is SplitLayout:
            output_file(self.id + ".html", title=self.id)
            distribution_plots = []
            for i in range(0, len(self.distribution_plots[list(self.distribution_plots.keys())[0]])):
                for key in self.distribution_plots.keys():
                    distribution_plots.append(self.distribution_plots[key][i])
            show(gridplot(distribution_plots, ncols=self.layout.ncols, plot_width=self.layout.plot_width, plot_height=self.layout.plot_eight, toolbar_location=None))

class Layout:
    def __init__(self, width):
        self.width = width
        self.plot_eight = 400
        self.plot_width = width
        self.plot_resolution = width
        self.ncols = 1

# Plot all graphs on the same page on a single column.
class VerticalLayout(Layout):
    def __init__(self, width):
        super().__init__(width)

# Plot different graph to different pages.
class SeparateLayout(Layout):
    def __init__(self, width):
        super().__init__(width)

# Plot different categories graphs with same index as tuples on the same line.
class SplitLayout(Layout):
    def __init__(self, width, n):
        super().__init__(width)
        self.plot_width = math.floor(width / n)
        self.plot_resolution = math.floor(width / n)
        self.ncols = n