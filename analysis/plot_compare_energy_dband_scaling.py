"""Plot the energies against the d-band centres from different sources."""

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

colors = {'ontop':'tab:red', 'threefold':'tab:blue'}

if __name__ == '__main__':
    """Compare the d-band centre from the LMTO calculations
    of Ruban et al. (1997), Vojvodic (2014) and my own."""

    # Remove the following metals
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    # The group that is chosen for the analysis
    CHOSEN_GROUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    # Adsorbates to be considered
    ADSORBATES = ['O', 'C']
    # Plot related
    color_row = ['tab:red', 'tab:blue', 'tab:green',]

    # Get the adsorption energies 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{CHOSEN_GROUP['energy']}.json"))
    # Get the site dependence of the energies
    data_from_site_dependence = json.load(open(f"output/adsorption_energies_{CHOSEN_GROUP['sampled']}.json"))

    # Plot figures for the energy vs. d-band centre for the different adsorbates
    # and choices of d-band centre 
    fig, ax = plt.subplots(2, 3, figsize=(5.5, 3), constrained_layout=True,
                           sharex=True, sharey=True)

    # Read in yaml files with the right quantities
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    data_from_vojvodic = yaml.safe_load(open('inputs/vojvodic_parameters.yaml'))

    with open(f"output/pdos_moments_{CHOSEN_GROUP['dos']}.json", 'r') as f:
        d_band_centres = json.load(f)

    collected_data = defaultdict(list)
    for ads_i, adsorbate in enumerate(ADSORBATES):
        texts = defaultdict(list) 
        for row_i, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            for metal in metal_row:

                if metal in REMOVE_LIST:
                    continue
                    
                # Get the energies
                energies = data_from_energy_calculation[adsorbate][metal]

                if isinstance(energies, list):
                    energies = np.min(energies)
                
                # sampled energies
                sampled_energies = data_from_site_dependence[adsorbate][metal]

                # Get the d-band centre
                d_band_centre = d_band_centres[metal]['d_band_centre']

                ax[ads_i,row_i].plot(d_band_centre, energies, 'o', color=colors['ontop'])

                ax[ads_i,row_i].plot(d_band_centre, np.min(sampled_energies), 'o',
                                    color=colors['threefold'])

                # Store texts to be adjusted later
                texts[row_i].append(ax[ads_i,row_i].text(d_band_centre, energies, 
                                    metal, fontsize=8, color='k'))

            # Adjust text for the different rows
            adjust_text(texts[row_i], ax=ax[ads_i,row_i], )

    for site, color in colors.items():
        ax[0,0].plot([] , [], 'o', color=color, label=site)
    ax[0,0].legend(loc='best', fontsize=7)

    for a in ax[1,:]:
        a.set_xlabel('$\epsilon_d$ (eV)')
    for i, a in enumerate(ax[:,0]):
        a.set_ylabel(r'$\Delta E$ for '+f'{ADSORBATES[i]}*')
    for i, a in enumerate(ax[0,:]):
        a.set_title(f'{i+3}$d$')
    

    fig.savefig('output/compare_energy_dband_centre_scaling.png', dpi=300)
    