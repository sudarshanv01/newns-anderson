"""Plot the damping function and the d-band centre relatiobnship."""

import numpy as np
import scipy
from scipy import special
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()

def damping(eps_d, b, eps_a):
    """Return the damping function."""
    return b / ( eps_d ) 

def final_fitting_function(eps_d, a, b, c, eps_a):
    """Return the final fitting function."""
    return a * damping(eps_d, b, eps_a) * eps_d +  c 

if __name__ == "__main__":
    """Plot the final fitting function."""

    eps_d_range = np.linspace(-4, -1, 100)
    a = -1
    b_range = [-1, -2, -3, -4]
    c = 0
    eps_a = -1

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    # get a cycle of colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, b in enumerate(b_range):
        ax.plot(eps_d_range, final_fitting_function(eps_d_range, a, b, c, eps_a), '-', label=r'$b = {}$'.format(b), color=colors[i])
        # Plot the dampling function
        ax.plot(eps_d_range, damping(eps_d_range, b, eps_a), '--', color=colors[i])
    
    ax.set_ylabel(r'$\Delta E$ (eV)')
    ax.set_xlabel(r'$\epsilon_d$ (eV)')
    ax.legend(loc='best')

    fig.savefig('output/testing_damping_function.png', dpi=300)

