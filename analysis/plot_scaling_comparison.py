
"""Compare scaling of O* and C* with the model and the DFT result"""
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from adjustText import adjust_text
from plot_params import get_plot_params
from plot_fitting import JensNewnsAnderson
import numpy as np
from pprint import pprint
get_plot_params()
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

if __name__ == '__main__':
    """Compare scaling from the model and from DFT."""
    FUNCTIONAL = 'PBE_scf'
    REMOVE_LIST = []
    C_eps_a = -1
    O_eps_a = -5

    # Read the energy file.
    with open(f'output/adsorption_energies_{FUNCTIONAL}.json', 'r') as f:
        ads_energy = json.load(f)

    # Read the pdos file
    with open(f'output/pdos_moments_{FUNCTIONAL}.json', 'r') as f:
        pdos_data = json.load(f)
    
    # Read data from LMTO.
    with open(f"inputs/data_from_LMTO.json", 'r') as f:
        data_from_LMTO = json.load(f)

    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{FUNCTIONAL}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{FUNCTIONAL}.json", 'r') as f:
        c_parameters = json.load(f)

    # Parameters for the idealised model.
    eps_d_range = np.linspace(-5, 0, 100)

    # Collect the minimum and maximum Vsd and width values
    # for each row
    vsd_all = defaultdict(list)
    width_all = defaultdict(list)
    for i, row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        for metal in row:
            try:
                width = pdos_data[metal]['width'] 
            except KeyError:
                continue
            Vsd = np.sqrt(data_from_LMTO['Vsdsq'][metal])
            width_all[i].append(width)
            vsd_all[i].append(Vsd)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    texts = defaultdict(list)
    for metal in ads_energy['C']:
        if metal in REMOVE_LIST:
            continue
        if metal not in ads_energy['O']:
            continue
        if metal not in pdos_data:
            continue
        if metal in FIRST_ROW:
            color = 'tab:red'
            index = FIRST_ROW.index(metal)
        elif metal in SECOND_ROW:
            color = 'tab:orange'
            index = SECOND_ROW.index(metal)
        elif metal in THIRD_ROW:
            color = 'tab:green'
            index = THIRD_ROW.index(metal)
        
        # Plot scaling between adsorbates
        ax.plot(ads_energy['C'][metal], ads_energy['O'][metal],'o', color=color)
        texts[0].append(ax.text(ads_energy['C'][metal], ads_energy['O'][metal], metal, color=color, fontsize=12))

    # Plot the idealised scaling for the model
    for i, row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        
        if i == 0:
            color = 'tab:red'
        elif i == 1:
            color = 'tab:orange'
        elif i == 2:
            color = 'tab:green'

        c_energies = []
        o_energies = []

        # min_vsd = np.min(vsd_all[i])
        # max_vsd = np.max(vsd_all[i])
        # min_width = np.min(width_all[i])
        # max_width = np.max(width_all[i])

        if i == 0:
            min_vsd = 0.5
            max_vsd = 1.5
            min_width = 3.5
            max_width = 4.5
        elif i == 1:
            min_vsd = 1.5
            max_vsd = 2.5
            min_width = 4.5
            max_width = 5.5
        elif i == 2:
            min_vsd = 2.5
            max_vsd = 3.5
            min_width = 5.5
            max_width = 6.5

        # get the idealised energies for the different metals
        energies_model = defaultdict(list)
        for metal in row:
            try:
                eps_d = pdos_data[metal]['d_band_centre']
            except KeyError:
                continue
            filling = data_from_LMTO['filling'][metal]

            energies_c_minmax = []
            energies_o_minmax = []

            for vsd, width in [(min_vsd, min_width), (max_vsd, max_width)]:
                c_jna = JensNewnsAnderson(
                    [vsd],
                    [filling],
                    [width,],
                    C_eps_a,
                )
                o_jna = JensNewnsAnderson(
                    [vsd],
                    [filling],
                    [width,],
                    O_eps_a,
                )

                energy_c = c_jna.fit_parameters([eps_d], **c_parameters)
                energy_o = o_jna.fit_parameters([eps_d], **o_parameters)

                energies_c_minmax.append(energy_c[0])
                energies_o_minmax.append(energy_o[0])

            energies_model['O'].append(energies_o_minmax)
            energies_model['C'].append(energies_c_minmax)

        # Plot fill betweens of minimum and maximum energies of the model
        min_energies_c, max_energies_c = np.array(energies_model['C']).T
        min_energies_o, max_energies_o = np.array(energies_model['O']).T

        ax.plot(min_energies_c, min_energies_o, color=color, linestyle='--')
        ax.plot(max_energies_c, max_energies_o, color=color, linestyle='--')

    for i, text in texts.items():
        adjust_text(text, ax=ax)
    
    ax.set_xlabel(r'$\Delta E_{\rm C}$ (eV)')
    ax.set_ylabel(r'$\Delta E_{\rm O}$ (eV)')

    fig.savefig(f'output/scaling_comparison_{FUNCTIONAL}.png')
