"""Plot the scaling relations from the DFT energies."""
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from adjustText import adjust_text
from plot_params import get_plot_params
get_plot_params()
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 
if __name__ == '__main__':
    """Plot the scaling relations from the energy file."""
    FUNCTIONAL = 'PBE_fixed'
    REMOVE_LIST = ['Ti', 'V', 'Cr', 'Re', 'Ru', 'Os']
    # Read the energy file.
    with open(f'output/adsorption_energies_{FUNCTIONAL}.json', 'r') as f:
        ads_energy = json.load(f)

    # Read the pdos file
    with open(f'output/pdos_moments_{FUNCTIONAL}.json', 'r') as f:
        pdos_data = json.load(f)

    # Plot the variation of energies with the d-band center and 
    # scaling with themselves.
    fig, ax = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
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

        # Plot the scaling of C with the d-band center.
        ax[0].plot(pdos_data[metal]['d_band_centre'], ads_energy['C'][metal], 'o', color=color)
        texts[0].append(ax[0].text(pdos_data[metal]['d_band_centre'], ads_energy['C'][metal], metal, color=color, fontsize=12))

        # Plot the scaling of O with the d-band center.
        ax[1].plot(pdos_data[metal]['d_band_centre'], ads_energy['O'][metal], 'o', color=color)
        texts[1].append(ax[1].text(pdos_data[metal]['d_band_centre'], ads_energy['O'][metal], metal, color=color, fontsize=12))

        # Plot scaling between adsorbates
        ax[2].plot(ads_energy['C'][metal], ads_energy['O'][metal],'o', color=color)
        texts[2].append(ax[2].text(ads_energy['C'][metal], ads_energy['O'][metal], metal, color=color, fontsize=12))

    for i, text in texts.items():
        adjust_text(text, ax=ax[i])

    # Set the labels.
    ax[0].set_xlabel(r'$\epsilon_d$ (eV)')
    ax[0].set_ylabel(r'$\Delta E_{\rm C}$ (eV)')
    ax[1].set_xlabel(r'$\epsilon_d$ (eV)')
    ax[1].set_ylabel(r'$\Delta E_{\rm O}$ (eV)')

    ax[2].set_xlabel(r'$\Delta E_{\rm C}$ (eV)')
    ax[2].set_ylabel(r'$\Delta E_{\rm O}$ (eV)')

    fig.savefig(f'output/adsorption_energies_{FUNCTIONAL}.png')