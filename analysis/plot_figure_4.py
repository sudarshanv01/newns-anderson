"""Plot an idealised scaling model for C and O scaling."""
import numpy as np
import json
from plot_fitting import JensNewnsAnderson
from collections import defaultdict
import matplotlib.pyplot as plt
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 
if __name__ == '__main__':
    """Plot a scaling line with O* and C* based on the fit parameters from DFT."""
    FUNCTIONAL = 'PBE_scf'
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{FUNCTIONAL}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{FUNCTIONAL}.json", 'r') as f:
        c_parameters = json.load(f)

    # Plot everything over a range of d-band values
    eps_d_range = np.linspace(-4, 0, 20)
    # Assume that the filling of the d-band varies linearly from 1. to 0
    # in the same interval
    filling_d_range = np.linspace(1, 0, 20)
    # Choose a set of constant parameters.
    METALS = [FIRST_ROW, SECOND_ROW, THIRD_ROW]
    parameters = [ [1, 4], [2, 5], [3, 6] ]
    ADSORBATES = {'C': -1, 'O': -5}

    energies = defaultdict(lambda: defaultdict(list))

    for adsorbate, eps_a in ADSORBATES.items():
        for i, row in enumerate(METALS):
            # get the hybridisation energies
            jna = JensNewnsAnderson( Vsd = parameters[i][0]*np.ones(len(eps_d_range)),
                                     filling = filling_d_range,
                                     eps_a = eps_a,
                                     width = parameters[i][1]*np.ones(len(eps_d_range)))
            if adsorbate == 'O':
                jna.fit_parameters(eps_d_range, **o_parameters)
            elif adsorbate == 'C':
                jna.fit_parameters(eps_d_range, **c_parameters)
            else:
                raise ValueError(f"Unknown adsorbate {adsorbate}")

            energy = jna.hybridisation_energy
            energies[adsorbate][i] = energy

    # Plot the adsorbate energy against each other  
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    ax.scatter(energies['C'][0], energies['O'][0], marker='*', c=eps_d_range, label='First row', cmap='coolwarm')
    ax.scatter(energies['C'][1], energies['O'][1], marker='s', c=eps_d_range, label='Second row', cmap='coolwarm')
    cax = ax.scatter(energies['C'][2], energies['O'][2], marker='o', c=eps_d_range, label='Third row', cmap='coolwarm')
    # Plot colormap
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.set_ylabel('$\epsilon_d$ (eV)')

    ax.set_xlabel('Hybridisation C* (eV)')
    ax.set_ylabel('Hybridisation O* (eV)')
    ax.legend(loc='best')

    fig.savefig('output/idealised_scaling_parameters.png')
                               

