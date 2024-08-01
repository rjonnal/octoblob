from matplotlib import pyplot as plt


def setup():
    plt.rcParams['font.size'] = 16
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

    # PPT slide is 13.3 width x 7.5 height, so default
    # size is most of the slide, leaving room for a title
    # at the top and some text at the right/left; this can
    # be overridden by specifying when calling plt.figure
    plt.rcParams['figure.figsize'] = (9,6)
    
    plt.rcParams['savefig.dpi'] = 100

def save(filename,figure_handle=None):
    if figure_handle is None:
        figure_handle = plt.gcf()
    figure_handle.savefig(filename)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

