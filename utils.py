import csv
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np


def load_dataset(pathname:str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset in csv format.

    Each line of the csv file represents a data from our dataset and each
    column represents the parameters.
    The last column corresponds to the label associated with our data.

    Parameters
    ----------
    pathname : str
        The path of the csv file.

    Returns
    -------
    data : ndarray
        All data in the database.
    labels : ndarray
        Labels associated with the data.
    """
    # check the file format through its extension
    if pathname[-4:] != '.csv':
        raise OSError("The dataset must be in csv format")
    # open the file in read mode
    with open(pathname, 'r') as csvfile:
        # create the reader object in order to parse the data file
        reader = csv.reader(csvfile, delimiter=',')
        # extract the data and the associated label
        # (he last column of the file corresponds to the label)
        data = []
        labels = []
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
        # converts Python lists into NumPy matrices
        # in the case of the list of labels, generate an int id per class
        data = np.array(data, dtype=np.float)
        lookupTable, labels = np.unique(labels, return_inverse=True)
    # return data with the associated label
    return data, labels


def plot_scatter_hist(data:np.ndarray, labels:np.ndarray, bins:int=30, figure:int=1) -> None:
    # fonctionne seulement sur des points 2D
    # Affiche un historgramme d'une liste de valeur et superpose l'affichage de la loi normale associée
    customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', '#04AF00', '#39A7FF',
                     '#7519CC', '#79E7FF', '#1863C15', '#B72EB9', '#EC2328', '#C86D39']
    fig = plt.figure(figure, clear=True)
    gridspec = fig.add_gridspec(ncols=2, nrows=2,
                                width_ratios=[3, 1],
                                height_ratios=[1, 3],
                                wspace=0.05,
                                hspace=0.05)
    
    # configuration of 2D point display
    scatter_axes = fig.add_subplot(gridspec[1, 0])
    scatter_axes.xaxis.set_minor_locator(AutoMinorLocator(2))
    scatter_axes.yaxis.set_minor_locator(AutoMinorLocator(2))
    scatter_axes.grid(b=True, which='both', linestyle='--')

    # configuration of the histogram display on the x component
    x_hist_axes = fig.add_subplot(gridspec[0, 0], sharex=scatter_axes, yticks=[])
    x_hist_axes.grid(b=True, which='both', linestyle='--')
    plt.setp(x_hist_axes.get_xticklabels(), visible=False)

    # configuration of the histogram display on the y component
    y_hist_axes = fig.add_subplot(gridspec[1, 1], sharey=scatter_axes, xticks=[])
    y_hist_axes.grid(b=True, which='both', linestyle='--')
    plt.setp(y_hist_axes.get_yticklabels(), visible=False)

    for y in np.unique(labels):
        # get the data associated with the label y
        x = data[labels == y]

        # display the 2D points
        scatter_axes.scatter(x=x[:,0],
                             y=x[:,1],
                             alpha=0.20,
                             color=customPalette[int(y)])
        # add the associated label
        scatter_axes.annotate(int(y),
                              x.mean(0),
                              horizontalalignment='center',
                              verticalalignment='center',
                              size=20, weight='bold',
                              color=customPalette[int(y)])
        
        # display of the histogram of the coordinates on x
        _, x_bins, _ = x_hist_axes.hist(x[:,0], bins=bins, density=True,
                                        alpha=0.40, color=customPalette[int(y)],
                                        orientation='vertical')
        x_mu = x[:,0].mean()
        x_sigma = x[:,0].std()
        x_pdf = ((1 / (np.sqrt(2 * np.pi) * x_sigma)) * np.exp(-0.5 * (1 / x_sigma * (x_bins - x_mu))**2))
        x_hist_axes.plot(x_bins, x_pdf, linewidth=2, color=customPalette[int(y)])

        # display of the histogram of the coordinates on y
        _, y_bins, _ = y_hist_axes.hist(x[:,1], bins=bins, density=True,
                                        alpha=0.40, color=customPalette[int(y)],
                                        orientation='horizontal')
        y_mu = x[:,1].mean()
        y_sigma = x[:,1].std()
        y_pdf = ((1 / (np.sqrt(2 * np.pi) * y_sigma)) * np.exp(-0.5 * (1 / y_sigma * (y_bins - y_mu))**2))
        y_hist_axes.plot(y_pdf, y_bins, linewidth=2, color=customPalette[int(y)])
    
    plt.pause(0.25)
