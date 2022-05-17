"""Plot the density of states with Al."""
import json
import matplotlib.pyplot as plt
import numpy as np
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':
    """Plot the density of states for difference
    adsorbates on Al."""

    # Read in the JSON file from the output filder
    with open('output/pdos_reference_Al.json', 'r') as handle:
        data = json.load(handle)
    
    METAL = 'Al'

    data.pop('H')

    fig, ax = plt.subplots(1, len(data), figsize=(6.25, 2.5), constrained_layout=True)

    # Plot each adsorbate separately
    for i, adsorbate in enumerate(data):
        energies, pdos = data[adsorbate][METAL]

        ax[i].plot(pdos, energies, '-')
        ax[i].set_title(adsorbate)

        ax[i].set_ylim([-10, 10])
        ax[i].set_xticks([])
    
    # Set the x-axis label
    # ax[0].set_xlabel('Density of States')
    # Set the y-axis label
    ax[0].set_ylabel('$\epsilon - \epsilon_F$ (eV)')
    fig.supxlabel('Projected Density of States (arb. units.)')
    fig.savefig('output/pdos_Al.png', dpi=300)




