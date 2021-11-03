"""Plot the projected density of states from the DFT calculations."""
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from ase.dft import get_distribution_moment
import numpy as np
from scipy.integrate import simps
from plot_params import get_plot_params
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 

def moment_generator(energies, all_dos, moment):
    """Get the moment of a distribution from the density of states"""
    def integrand_numerator(e, rho):
        return e * rho
    def integrand_denominator(e,rho):
        return rho
    def integrand_moment_numerator(e, epsilon_d, moment, rho):
        return ( (e - epsilon_d ) ** moment ) * rho
    moments = []
    eps_d = []
    energies = energies
    for alldos in all_dos:
        dos = np.array(alldos)
        epsilon_d_num = simps(integrand_numerator(energies,dos), energies)
        epsilon_d_den = simps(integrand_denominator(energies,dos), energies)
        epsilon_d = epsilon_d_num / epsilon_d_den
        moment_numerator = simps( integrand_moment_numerator(energies, epsilon_d, moment, \
                dos), energies)
        moment_denom = epsilon_d_den
        moment_spin = moment_numerator / moment_denom
        moments.append(moment_spin)
        eps_d.append(epsilon_d)
    return moments, eps_d

if __name__ == "__main__":
    """Plot the pdos for the metal and of the adsorbates from a DFT calculation."""
    FUNCTIONAL = 'PBE_scf'
    with open(f'output/pdos_{FUNCTIONAL}.json', 'r') as handle:
        pdos_data = json.load(handle)
    METALS = [FIRST_ROW, SECOND_ROW, THIRD_ROW]

    fig, ax = plt.subplots(len(METALS), len(METALS[0]), figsize=(16,12), sharey=True, constrained_layout=True)
    moments = defaultdict(dict)

    used_ij = []

    for metal in pdos_data['slab']:
        energies, pdos = pdos_data['slab'][metal]
        if metal in METALS[0]:
            i = 0
            j = METALS[0].index(metal)
        elif metal in METALS[1]:
            i = 1
            j = METALS[1].index(metal)
        elif metal in METALS[2]:
            i = 2
            j = METALS[2].index(metal)
        else:
            raise ValueError('Metal not in chosen list of metals.')

        used_ij.append((i,j))

        # make pdos and energies into numpy arrays
        energies = np.array(energies)
        if len(pdos) == 2:
            pdos = np.array(pdos).sum(axis=0)
        else:
            pdos = np.array(pdos)

        ax[i,j].plot(pdos, energies)
        ax[i,j].set_title(metal)
        ax[i,j].set_xticks([])
        ax[i,j].set_ylim(-10, 5)

        # get the d-band center and the width
        second_moment, center = moment_generator(energies, [ pdos ], 2)
        second_moment = second_moment[0]
        center = center[0]

        print('{} band center: {}'.format(metal, center))
        print('{} band second moment: {}'.format(metal, second_moment))
        ax[i,j].axhline(y=center, color='r', linestyle='--')

        moments[metal]['d_band_centre'] = center# [0]
        moments[metal]['width'] = 4 * np.sqrt(second_moment) 

        ax[i,j].axhline(y=center + 2 * np.sqrt(second_moment), color='g', linestyle='--')
        ax[i,j].axhline(y=center - 2 * np.sqrt(second_moment), color='g', linestyle='--')
        ax[i,j].axhline(y=0, color='k', linestyle='--')
    
    for i in range(len(METALS)):
        for j in range(len(METALS[i])):
            if (i,j) not in used_ij:
                fig.delaxes(ax[i,j])

    fig.savefig(f'output/pdos_{FUNCTIONAL}.png')


    with open(f'output/pdos_moments_{FUNCTIONAL}.json', 'w') as handle:
        json.dump(moments, handle, indent=4)

        