from matplotlib import pyplot as plt


def setup():
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.titlesize'] = 'medium'
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['figure.labelsize'] = 'medium'

    # turn off spines (plot border)
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False

    # this is for display while making the plot, not
    # saving; savefig.dpi governs saving (see below)
    plt.rcParams['figure.dpi'] = 100

    # Letter paper is 8.5 width x 11 height, so default
    # size is 6.5 (1 inch margins) x 4
    # this can be overridden by specifying when
    # calling plt.figure
    plt.rcParams['figure.figsize'] = (6.5,4)

    # For print we need high dpi; 300 minimum, up to 600
    plt.rcParams['savefig.dpi'] = 300

def save(filename,figure_handle=None):
    if figure_handle is None:
        figure_handle = plt.gcf()
    figure_handle.savefig(filename)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

