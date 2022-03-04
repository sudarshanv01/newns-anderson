"""Compare the projected density of states for different computational settings."""
import json
import yaml
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from collections import defaultdict
import numpy as np
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

if __name__ == '__main__':
    """Compare the d-density of states of the transition metals
    p-projected density of states."""

    # Remove the following metals
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    # The group that is chosen for the analysis
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))

    # Metals list
    METALS = [FIRST_ROW, SECOND_ROW, THIRD_ROW]

    # Generate a color list
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Data to compare, these will be separate plots
    data_to_compare = {
        # Compare the effect of relaxation
        'relaxation': ['PBE/SSSP_precision/gauss_smearing_0.1eV/dos_scf',
                       'PBE/SSSP_precision/gauss_smearing_0.1eV/relax',
                       ],

        # To ensure that the number of bands are converged 
        'system_size': ['PBE/SSSP_precision/gauss_smearing_0.1eV/dos_scf_small',
                        'PBE/SSSP_precision/gauss_smearing_0.1eV/dos_scf',
        ]
    }

    labels = {
        'relaxation': ['scf', 'relaxed'],
        'system_size': ['small', 'large']
    }

    # Decide what to compare, can be either energies, pdos or both
    compare_data = {'relaxation': ['energy'], 'system_size': ['pdos']}

    # Loop over the different data to compare
    for data_type, compare in data_to_compare.items():

        # Decide what to compare
        comparison_over = compare_data[data_type]
        if 'energy' in comparison_over: plot_energy = True
        else: plot_energy = False
        if 'pdos' in comparison_over: plot_pdos = True
        else: plot_pdos = False

        if plot_energy:
            figec, axec = plt.subplots(1, 2,
                                       figsize=(6.7, 3.5),
                                       constrained_layout=True)
        if plot_pdos:
            fig, ax = plt.subplots(len(METALS), len(METALS[1]), 
                                    figsize=(13,12), sharey=True,
                                    constrained_layout=True)

        # Store the energies for each dataset
        chem_energies_case = defaultdict(list) 

        # Get the chosen value of the d-band centre
        data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 

        # Iterate through cases and plot / store information
        for i, case in enumerate(compare):
            # Label for the group
            label_pdos = case.replace('/', '_')

            if plot_energy:    

                with open(f'output/adsorption_energies_{label_pdos}.json') as handle:
                    chem_energies = json.load(handle)

                for metal in chem_energies['C']:
                    if metal in REMOVE_LIST:
                        continue
                    d_band_centre = data_from_dos_calculation[metal]['d_band_centre'] 
                    # Get the chemisorption energy
                    chem_energies_case[label_pdos].append([ chem_energies['O'][metal],
                                                            chem_energies['C'][metal],
                                                            d_band_centre ] )

            if plot_pdos:
                # Read in the projected density of states
                with open(f'output/pdos_{label_pdos}.json') as handle:
                    pdos_data = json.load(handle)
                # Read in the pdos moments
                with open(f'output/pdos_moments_{label_pdos}.json') as handle:
                    pdos_moments = json.load(handle)

                # Keep track of which elements are computed
                used_ij = []
                color = COLORS[i]

                for metal in pdos_data['slab']:

                    # Do not plot metals in the remove list
                    if metal in REMOVE_LIST:
                        continue

                    # Get all the pdos
                    energies, pdos, pdos_sp = pdos_data['slab'][metal]

                    # Decide on the index based on the metal
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

                    # Plot the respective positions of the pdos
                    ax[i,j].plot(pdos, energies, color=color, alpha=0.5, lw=3)

                    # Axes related
                    used_ij.append((i,j))
                    ax[i,j].set_title(metal)
                    ax[i,j].set_xticks([])
                    ax[i,j].set_ylim(-12, 10)

                if j == 0:
                    ax[i,j].set_ylabel('$\epsilon - \epsilon_f$ (eV)')

        if plot_energy:
            # Plot the chemisorption energies of the different
            # computational methods against each other.
            # Plot against the d-band centre of the metal.
            for i, label_case in enumerate(chem_energies_case):
                energies_o, energies_c, d_band_centre = np.array(chem_energies_case[label_case]).T
                axec[0].plot(d_band_centre, energies_o, 'o', color=COLORS[i], label=labels[data_type][i], alpha=0.5)
                axec[1].plot(d_band_centre, energies_c, 'o', color=COLORS[i], alpha=0.5)

        if plot_pdos:
            ax[0,2].legend(loc='best')
            fig.savefig(f'output/pdos_compare_{data_type}.png')

        if plot_energy:
            axec[0].legend(loc='best')
            axec[0].set_ylabel(r'$\Delta E_{\mathrm{O}}$ (eV)')
            axec[1].set_ylabel(r'$\Delta E_{\mathrm{C}}$ (eV)')
            axec[0].set_xlabel(r'$\epsilon_d$ (eV)')
            axec[1].set_xlabel(r'$\epsilon_d$ (eV)')

            figec.savefig(f'output/chem_energies_compare_{data_type}.png', dpi=300)