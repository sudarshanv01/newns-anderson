"""Plot the d-band centres from different sources."""

import json
import yaml
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from collections import defaultdict
import numpy as np
from adjustText import adjust_text
get_plot_params()
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4
color_row = ['tab:red', 'tab:blue', 'tab:green',]

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

if __name__ == '__main__':
    """Compare the d-band centre from the LMTO calculations
    of Ruban et al. (1997), Vojvodic (2014) and my own."""

    # Remove the following metals
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    # The group that is chosen for the analysis
    CHOSEN_GROUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))['group'][0]

    # Plot figures for parity between my data and previous vales
    fig, ax = plt.subplots(1, 2, figsize=(6., 3), constrained_layout=True)

    # Read in yaml files with the right quantities
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    data_from_vojvodic = yaml.safe_load(open('inputs/vojvodic_parameters.yaml'))
    with open(f'output/pdos_moments_{CHOSEN_GROUP}.json', 'r') as f:
        d_band_centres = json.load(f)
    
    collected_data = defaultdict(list)
    for metal in d_band_centres:
        try:
            lmto_data = data_from_LMTO['d_band_centre'][metal]
            vojvodic_data = data_from_vojvodic['d_band_centre'][metal]
            own_data = d_band_centres[metal]['d_band_centre']
        except KeyError:
            continue
        collected_data['lmto'].append(lmto_data)
        collected_data['vojvodic'].append(vojvodic_data)
        collected_data['own'].append(own_data)
        collected_data['metal'].append(metal)

        if metal in FIRST_ROW:
            color = color_row[0]
        elif metal in SECOND_ROW:
            color = color_row[1]
        elif metal in THIRD_ROW:
            color = color_row[2]
        
        ax[0].plot(lmto_data, own_data, 'o', color=color, )
        ax[1].plot(vojvodic_data, own_data, 'o', color=color, )

    
    # Plot the data
    # ax[0].plot(collected_data['lmto'], collected_data['own'], 'o', color='#1f77b4')
    # ax[1].plot(collected_data['vojvodic'], collected_data['own'], 'o', color='#ff7f0e')

    ax[0].set_xlabel('$\epsilon_d$ (eV) LMTO Ruban et al. (1997)')
    ax[1].set_xlabel('$\epsilon_d$ (eV) Vojvodic et al. (2014)')
    ax[0].set_ylabel('$\epsilon_d$ (eV) Quantum ESPRESSO - PW')

    # Plot a parity line in both axes
    ax[0].plot(ax[0].get_xlim(), ax[0].get_xlim(), '--', color='k')
    ax[1].plot(ax[1].get_xlim(), ax[1].get_xlim(), '--', color='k')

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    for i, color in enumerate(color_row):
        ax[0].plot([], [], color=color, label=f'{i+3}$d$')
    ax[0].legend(loc='best', fontsize=8)

    fig.savefig('output/compare_dband_centre.png', dpi=300)
    