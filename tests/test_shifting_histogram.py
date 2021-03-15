from octoblob import bin_shift_histogram
import numpy as np
from matplotlib import pyplot as plt


def make_distribution(N,phase_mean,phase_standard_deviation):
    # make a phase distribution with given mean and standard deviation,
    # with N samples

    # create a normal distribution with the right std
    vals = np.random.randn(N)*phase_standard_deviation

    # add the phase mean
    vals = vals + phase_mean

    # to wrap this using modulo, we have to add pi, compute
    # the mod 2pi, and then subtract the pi:
    vals = vals + np.pi 
    vals = vals%(2*np.pi) 
    vals = vals - np.pi

    return vals

phase_mean = 2.5 # radians
phase_standard_deviation = 5.1*np.pi
N = 1000

dist = make_distribution(N,phase_mean,phase_standard_deviation)


n_bins = 10
bin_edges = np.linspace(-np.pi,np.pi,n_bins+1)
bin_centers = bin_edges[:-1]+np.mean(np.diff(bin_edges))/2.0

# use numpy.histogram to make a histogram, using bin_edges
hist,_ = np.histogram(dist)
plt.figure()
plt.bar(bin_centers,hist)
plt.title('original histogram')

for resample_factor in [2,4,8,16,32]:
    counts,centers = bin_shift_histogram(dist,bin_centers,resample_factor)
    plt.figure()
    plt.bar(centers,counts)
    plt.title('resample factor %d'%resample_factor)
    
plt.show()

