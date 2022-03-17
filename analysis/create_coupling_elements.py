"""Create coupling elements for the Newns-Anderson analysis."""
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ase.data import covalent_radii, atomic_numbers
from plot_params import get_plot_params
get_plot_params()
import matplotlib as mpl
from ase import units
mpl.rcParams['lines.markersize'] = 4

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 

def create_coupling_elements(s_metal, s_Cu, 
    anderson_band_width, anderson_band_width_Cu, 
    r=None, r_Cu=None, normalise_bond_length=False,
    normalise_by_Cu=True):
    """Create the coupling elements based on the Vsd
    and r values. The Vsd values are identical to those
    used in Ruban et al. The assume that the bond lengths
    between the metal and adsorbate are the same. Everything
    is referenced to Cu, as in the paper by Ruban et al."""
    Vsdsq = s_metal**5 * anderson_band_width
    Vsdsq_Cu = s_Cu**5 * anderson_band_width_Cu 
    if normalise_by_Cu:
        Vsdsq /= Vsdsq_Cu
    if normalise_bond_length:
        assert r is not None
        if normalise_by_Cu: 
            assert r_Cu is not None
            Vsdsq *= r_Cu**8 / r**8
        else:
            Vsdsq /= r**8
    return Vsdsq


if __name__ == '__main__':
    """Create the coupling elements relative to Cu using
    the same methodology as in Ruban et al. 
    (https://doi.org/10.1016/S1381-1169(96)00348-2) 
    using the band with from GGA-DFT calculations and compare
    it with values taken from Anderson, Jensen, Gotzel."""

    # Plot the Vsd terms, only plot for each row of TMs
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 2), constrained_layout=True, sharex=True, sharey=True)

    # Load the input quantities from the LMTO calculation
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Load data from the DFT calculation
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 

    # Remove list
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']

    Vsdsq_data = data_from_LMTO['Vsdsq']
    filling_data = data_from_LMTO['filling']
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']

    coupling_elements = defaultdict(list)

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
            tabulated_Vsdsq = create_coupling_elements(s_metal=s_data[metal],
                s_Cu=s_data['Cu'],
                anderson_band_width=anderson_band_width_data[metal],
                anderson_band_width_Cu=anderson_band_width_data['Cu'],
                normalise_bond_length=False,
                normalise_by_Cu=True)

            # Create the bond length based on the Wigner-Seitz
            # radii and the covalent radii of the atoms
            bond_length = s_data[metal]*units.Bohr # \
                        # + covalent_radii[atomic_numbers[adsorbate]] 
            bond_length_Cu = s_data['Cu']*units.Bohr #\
                        # + covalent_radii[atomic_numbers[adsorbate]]

            Vsdsq_rdep = create_coupling_elements(s_metal=s_data[metal],
                s_Cu=s_data['Cu'],
                anderson_band_width=anderson_band_width_data[metal],
                anderson_band_width_Cu=anderson_band_width_data['Cu'],
                r=bond_length,
                r_Cu=bond_length_Cu,
                normalise_bond_length=True,
                normalise_by_Cu=True)

            print(f'{metal} Vsdsq: {Vsdsq}, tabulated Vsdsq: {tabulated_Vsdsq}, DFT Vsdsq: {Vsdsq_rdep}')

            # Plot the data against the filling
            filling = filling_data[metal]
            ax[row_i].plot(filling, Vsdsq, 'v', color='tab:blue', alpha=0.5)
            ax[row_i].plot(filling, tabulated_Vsdsq, 'o', color='tab:orange', alpha=0.5)
            ax[row_i].plot(filling, Vsdsq_rdep, '*', color='tab:green', alpha=0.5)

            # Save the data
            coupling_elements[metal] = Vsdsq_rdep
    
    # Save the data
    json.dump(coupling_elements, open(f'output/dft_Vsdsq.json', 'w'), indent=4)

    for a in ax:
        a.set_xlabel('Filling')
    for i, a in enumerate(ax):
        a.set_ylabel(r'$V_{pd}^2 / V_{p,Cu}^2$')
    ax[0].plot([], [], 'o', color='tab:orange', label='LMTO')
    ax[0].plot([], [], 'o', color='tab:green', label='DFT')
    ax[0].legend(loc='best')

    fig.savefig(f'output/Vsdsq_comparison.png', dpi=300)
