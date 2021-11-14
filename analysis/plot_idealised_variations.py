"""Plot the idealised variation of the the total, hybridisation and ortho energy."""
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import JensNewnsAnderson

def create_plot_layout():
    """Create a plot layout for plotting the Newns-Anderson
    dos and the energies of orthogonalisation, spd hybridisation
    energy for each specific adsorbate."""
    fig = plt.figure(figsize=(14,12), constrained_layout=True)
    gs = fig.add_gridspec(nrows=12, ncols=3,)
    # The first 2 rows will be the orthogonalisation energies, spd
    # hybridisation energy and the total energy as a function of the
    # d-band centre.
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax2 = fig.add_subplot(gs[0:3, 1])
    ax3 = fig.add_subplot(gs[0:3, 2])
    # Set the axes labels
    ax1.set_xlabel('d-band centre (eV)')
    ax1.set_ylabel('NA energy (eV)')
    ax2.set_xlabel('d-band centre (eV)')
    ax2.set_ylabel('Ortho energy (eV)')
    ax3.set_xlabel('d-band centre (eV)')
    ax3.set_ylabel('Total energy (eV)')
    # Then make three plots with the density of states coming from
    # the different solutions of the Newns-Anderson equation.
    ax4 = fig.add_subplot(gs[3:, 0])
    ax5 = fig.add_subplot(gs[3:, 1])
    ax6 = fig.add_subplot(gs[3:, 2])
    # Set the axes labels
    ax4.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax5.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax6.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax4.set_ylabel('Projected Density of States')
    # Remove y-ticks from 4, 5, 6
    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])
    return fig, np.array([ [ax1, ax2, ax3], [ax4, ax5, ax6] ])


if __name__ == '__main__':
    """Create an idealised scaling plot of the chemisorption energies
    and the separate hybridisation and """