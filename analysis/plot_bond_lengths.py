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
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    # Store text for adjustable plots
    text_C = []
    text_O = []

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
                    ax[0].plot(filling_data[metal], bond_lengths['C'][metal], 'o', color=color)
                    text_C.append(ax[0].text(filling_data[metal], bond_lengths['C'][metal], metal, color=color, fontsize=15))
                if metal in bond_lengths['O']:
                    ax[1].plot(filling_data[metal], bond_lengths['O'][metal], 'o', color=color)
                    text_O.append(ax[1].text(filling_data[metal], bond_lengths['O'][metal], metal, color=color, fontsize=15))

    ax[0].set_xlabel('Filling')
    ax[0].set_ylabel('Bond length C* (Å)')
    ax[1].set_xlabel('Filling')
    ax[1].set_ylabel('Bond length O* (Å)')

    adjust_text(text_C, ax=ax[0])
    adjust_text(text_O, ax=ax[1])

    fig.savefig(f'output/bond_lengths.png')