"""Plot bond lengths from a DFT calculation."""
import json
from pprint import pprint
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
from adjustText import adjust_text
from plot_params import get_plot_params
import numpy as np
import yaml
from ase.data import covalent_radii, atomic_numbers
from ase import units
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 

if __name__ == "__main__":
    """Plot the bond length stored in all files with 
    the name outputs/bond_lengths_*.json. based on the
    metal row."""

    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    COMP_SETUP = COMP_SETUP['energy']
    with open("inputs/data_from_LMTO.json", "r") as handle:
        data_from_LMTO = json.load(handle)
    filling_data = data_from_LMTO['filling']

    # Get all files with the name bond_lengths_*.json
    files = glob.glob(f"output/bond_lengths_{COMP_SETUP}*.json")

    ADSORBATES = ['C', 'O']

    # Get data from file
    bond_lengths_all = defaultdict(dict) 

    for handle in files:
        type_calc = handle.split("_")[-1].split(".")[0]
        with open(handle) as f:
            data = json.load(f)
            for adsorbate in ADSORBATES:
                bond_lengths_all[type_calc][adsorbate] = data[adsorbate]

    pprint(bond_lengths_all) 

    # Plot the data according to the metal row
    fig, ax = plt.subplots(3, 2, figsize=(6., 7), constrained_layout=True)

    # Store text for adjustable plots
    text_C = []
    text_C2 = []
    text_O = []
    text_O2 = []

    for type_calc, bond_lengths in bond_lengths_all.items():
        for row_index, row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            if row_index == 0:
                color ='tab:blue'
            elif row_index == 1:
                color ='tab:orange'
            elif row_index == 2:
                color ='tab:green'

            for metal in row:
                if metal in bond_lengths['C']:
                    ax[0,0].plot(filling_data[metal], bond_lengths['C'][metal], 'o', color=color)
                    text_C.append(ax[0,0].text(filling_data[metal], bond_lengths['C'][metal], metal, color=color))
                    # Plot the bond length assuming it is the sum of the 
                    # Wigner-Seitz radius and the covalent radius
                    bond_length = data_from_LMTO['s'][metal]*units.Bohr #+ covalent_radii[atomic_numbers['C']] 
                    wigner_seitz_radii = data_from_LMTO['s'][metal]*units.Bohr
                    anderson_band_width = data_from_LMTO['anderson_band_width'][metal]
                    ax[1,0].plot(filling_data[metal], bond_length, 'v', color=color)
                    text_C2.append(ax[1,0].text(filling_data[metal], bond_length, metal, color=color))
                    # ax[2,0].plot(filling_data[metal], anderson_band_width, 'v', color=color)
                    ax[2,0].plot(filling_data[metal], wigner_seitz_radii**5, 'v', color=color)
                if metal in bond_lengths['O']:
                    ax[0,1].plot(filling_data[metal], bond_lengths['O'][metal], 'o', color=color)
                    text_O.append(ax[0,1].text(filling_data[metal], bond_lengths['O'][metal], metal, color=color))
                    anderson_band_width = data_from_LMTO['anderson_band_width'][metal]
                    # Plot the bond length assuming it is the sum of the 
                    # Wigner-Seitz radius and the covalent radius
                    bond_length = data_from_LMTO['s'][metal]*units.Bohr #+ covalent_radii[atomic_numbers['O']] 
                    ax[1,1].plot(filling_data[metal], bond_length, 'v', color=color)
                    text_O2.append(ax[1,1].text(filling_data[metal], bond_length, metal, color=color))
                    ax[2,1].plot(filling_data[metal], anderson_band_width, 'v', color=color)

    for a in ax[0,:]:
        a.set_xlabel('Filling')
        a.set_ylabel('Bond length C* (Å)')
    for a in ax[1,:]:
        a.set_xlabel('Filling')
        a.set_ylabel('Wigner-Seitz radius, s (Å)')
    for a in ax[2,:]:
        a.set_xlabel('Filling')
    ax[2,1].set_ylabel('Anderson band width (eV)')
    ax[2,0].set_ylabel('$s^{5}$ (Å$^5$)')
    fig.delaxes(ax[1,1])

    adjust_text(text_C, ax=ax[0,0])
    adjust_text(text_O, ax=ax[0,1])
    adjust_text(text_C2, ax=ax[1,0])
    adjust_text(text_O2, ax=ax[1,1])

    fig.savefig(f'output/bond_lengths.png', dpi=300)