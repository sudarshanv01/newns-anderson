"""Create coupling elements for the Newns-Anderson analysis."""
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from plot_params import get_plot_params
get_plot_params()
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 

if __name__ == '__main__':
    """Create the coupling elements relative to Cu using
    the same methodology as in Ruban et al. 
    (https://doi.org/10.1016/S1381-1169(96)00348-2) 
    using the band with from GGA-DFT calculations and compare
    it with values taken from Anderson, Jensen, Gotzel."""

    # Plot the Vsd terms, only plot for each row of TMs
    fig, ax = plt.subplots(2, 3, figsize=(5.5, 4), constrained_layout=True, sharex=True, sharey=True)

    # Load the input quantities from the LMTO calculation
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    # Load data from the DFT calculation
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 

    # Remove list
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']

    # Adsorbates studied
    ADSORBATES = ['O', 'C']

    Vsdsq_data = data_from_LMTO['Vsdsq']
    filling_data = data_from_LMTO['filling']
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    bond_lengths = json.load(open(f"output/bond_lengths_{COMP_SETUP['energy']}.json"))

    coupling_elements = defaultdict(dict)

    for ads_i, adsorbate in enumerate(ADSORBATES):
        for row_i, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            for metal in metal_row:

                if metal in REMOVE_LIST:
                    continue

                # Plot the Vsdsq from the LMTO calculation
                # and compare it with that in Jens' book
                # along with the Vsdsq computed from 
                # the GGA-DFT calculation, all Vsdsq values
                # referenced to that of Cu; assuming a constant
                # bond distance between the metal and the adsorbate

                # From the Textbook
                Vsdsq = Vsdsq_data[metal]

                # Calculated from Anderson, Jensen, Gotzel
                tabulated_Vsdsq = s_data[metal]**5 * anderson_band_width_data[metal]
                tabulated_Vsdsq /= ( s_data['Cu']**5 * anderson_band_width_data['Cu'] )

                # Now compute the same quantity assuming that Delta is the
                # band width from the DFT calculation
                bond_length = bond_lengths[adsorbate][metal]
                bond_length_Cu = bond_lengths[adsorbate]['Cu']
                dft_Vsdsq = s_data[metal]**5 * data_from_dos_calculation[metal]['width']
                dft_Vsdsq /= ( s_data[metal]**5 * data_from_dos_calculation['Cu']['width'] )
                dft_Vsdsq *= bond_length_Cu**8 / bond_length**8

                print(f'{metal} Vsdsq: {Vsdsq}, tabulated Vsdsq: {tabulated_Vsdsq}, DFT Vsdsq: {dft_Vsdsq}')

                # Plot the data against the filling
                filling = filling_data[metal]
                # ax[ads_i,row_i].plot(filling, Vsdsq, 'o', color='tab:blue', alpha=0.5)
                ax[ads_i,row_i].plot(filling, tabulated_Vsdsq, 'o', color='tab:orange')
                ax[ads_i,row_i].plot(filling, dft_Vsdsq, 'o', color='tab:green')

                # Save the data
                coupling_elements[adsorbate][metal] = dft_Vsdsq
    
    # Save the data
    json.dump(coupling_elements, open('output/dft_Vsdsq.json', 'w'), indent=4)

    for a in ax[1,:]:
        a.set_xlabel('Filling')
    for i, a in enumerate(ax[:,0]):
        a.set_ylabel(r'$V_{pd}^2 / V_{p,Cu}^2$ for '+f'{ADSORBATES[i]}*')
    ax[0,1].plot([], [], 'o', color='tab:orange', label='LMTO')
    ax[0,1].plot([], [], 'o', color='tab:green', label='DFT')
    ax[0,1].legend(loc='best')

    fig.savefig('output/Vsdsq_comparison.png', dpi=300)
