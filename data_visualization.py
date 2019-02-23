import numpy as np
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
import utilities

DISTRIBUTION_PLOT_WIDTH = 1250
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
    values = utilities.get(model, parameter_type)
    minimum = min(values)
    maximum = max(values)
    
    hist, edges = np.histogram(values, density=True, bins=resolution)
    x = np.linspace(minimum, maximum, resolution)

    plot = make_plot(title + " - " + parameter_type, hist, edges, x, parameter_type, "frequency")
    if parameter_type not in distribution_plots.keys():
        distribution_plots[parameter_type] = []
    distribution_plots[parameter_type].append(plot)

def output_plots():
    for key in distribution_plots.keys():
        output_file(key + '_distribution.html', title=key + " distribution")
        show(gridplot(distribution_plots[key], ncols=1, plot_width=DISTRIBUTION_PLOT_WIDTH, plot_height=400, toolbar_location=None))