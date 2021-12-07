"""Compare the coupling elements from Vsd, fitting and LCAO."""
import numpy as np
import json
from pprint import pprint
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
from adjustText import adjust_text
from plot_params import get_plot_params
from ase.data import covalent_radii, atomic_numbers
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 

if __name__ == "__main__":
    """Compare coupling elements coming from different sources
    1. Vsd
    2. Fitting procedure from Norskov-Newns-Anderson
    3. LCAO
    """

    # Functional to use for fitting data
    # FUNCTIONAL = 'PBE_scf'
    FUNCTIONAL = 'PBE_scf_cold_smearing_0.2eV'
    ADSORBATE = ['C', 'O']

    # Plot the inverse of bond lengths, Vsd, Vak and LCAO Vak
    fig, ax = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)

    # Read data from LMTO orbitals
    with open("inputs/data_from_LMTO.json", 'r') as handle:
        data_from_LMTO = json.load(handle)

    # Get data from LMTO orbitals 
    Vsdsq_data = data_from_LMTO["Vsdsq"]
    ideal_filling_data = data_from_LMTO["filling"]

    # Get all files with the name bond_lengths_*.json
    files = glob.glob(f"output/bond_lengths_{FUNCTIONAL}_*.json")

    # Get data from fitting
    with open(f"output/O_parameters_{FUNCTIONAL}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{FUNCTIONAL}.json", 'r') as f:
        c_parameters = json.load(f)
    fitting_parameters_energy = {'C': c_parameters, 'O': o_parameters}
    # fitting_parameters_dos = json.load(open(f"output/Vak_data_{FUNCTIONAL}.json", 'r'))
    
    # Get data from LCAO 
    with open(f"output/fit_results_lcao.json", 'r') as f:
        lcao_data = json.load(f)

    # Get data from file
    bond_lengths = {} 
    for handle in files:
        # adsorbate = handle.replace('_scf', '').split('/')[1].split('.')[0].split('_')[3]
        if 'C' in handle:
            adsorbate = 'C'
        elif 'O' in handle:
            adsorbate = 'O'
        with open(handle) as f:
            data = json.load(f)
            bond_lengths[adsorbate] = data

    for i, adsorbate in enumerate(ADSORBATE):
        # Iterate over row
        for row_index, row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            if row_index == 0:
                color ='tab:blue'
            elif row_index == 1:
                color ='tab:orange'
            elif row_index == 2:
                color ='tab:green'
            ax[i,0].plot([], [], color=color, label=f'{row_index+3}d')
            
            for index, metal in enumerate(row):
                # Filling - as an x-axis
                try:
                    filling = ideal_filling_data[metal]
                except KeyError:
                    continue
                try:
                    bond_length = bond_lengths[adsorbate][metal]
                    ax[i,0].plot(filling, bond_length, 'o', color=color)
                except KeyError:
                    pass
                try:
                    # Vsd from LMTO
                    Vsd = np.sqrt(Vsdsq_data[metal])
                    ax[i,1].plot(filling, Vsd, 'o', color=color)
                except KeyError:
                    pass
                try:
                    # Vak from fitting
                    Vak = fitting_parameters_energy[adsorbate]['Vak'][metal]
                    ax[i,2].plot(filling, Vak, 'o', color=color)
                except KeyError:
                    pass
                try:
                    # Vak from LCAO
                    Vak_lcao = lcao_data[adsorbate][metal]['Vak_calc']
                    ax[i,3].plot(filling, Vak_lcao, 'o', color=color)
                except KeyError:
                    pass


        ax[i,0].set_xlabel('Filling')
        ax[i,0].set_ylabel('Bond length (Å)')
        ax[i,1].set_xlabel('Filling')
        ax[i,1].set_ylabel('$V_{sd}/V_{sd}^{Cu}$ (LMTO)')
        ax[i,2].set_xlabel('Filling')
        ax[i,2].set_ylabel('$V_{ak}$ %s* (fit, energy)'%adsorbate)
        ax[i,3].set_xlabel('Filling')
        ax[i,3].set_ylabel('$V_{ak} = \sqrt{\sum V_i^2}$ (LCAO)')
        ax[i,0].legend(loc='best')


    fig.savefig(f'output/coupling_term_{FUNCTIONAL}.png')