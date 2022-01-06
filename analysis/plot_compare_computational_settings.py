"""Compare the projected density of states for different computational settings."""
import json
import yaml
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()

# Define periodic table of elements
SP_METALS   = [ 'Mg', 'Al' ]
FIRST_ROW   = [ 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'Os', 'Ir', 'Pt', 'Au',] 

if __name__ == '__main__':
    """Compare the d-density of states of the transition metals
    p-projected density of states for the adsorbate sp-projected
    density of states for Al and Mg and the energies."""

    # Remove the following metals
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    REMOVE_LIST.remove('Mg')
    REMOVE_LIST.remove('Al')

    # Metals list
    METALS = [SP_METALS, FIRST_ROW, SECOND_ROW, THIRD_ROW]

    # Generate a color list
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Data to compare, these will be separate plots
    data_to_compare = {
        # Compare the effect of relaxation
        'relaxation': ['PBE/SSSP_efficiency/cold_smearing_0.1eV/dos_relax',
                       'PBE/SSSP_efficiency/cold_smearing_0.1eV/dos_scf',
                       ],
        # Compare the effect of the smearing
        'smearing': ['PBE/SSSP_efficiency/cold_smearing_0.1eV/dos_scf',
                     'PBE/SSSP_efficiency/cold_smearing_0.2eV/dos_scf',
                     'PBE/SSSP_efficiency/tetrahedron_smearing/dos_scf',
                    ],
        # Compare the effect of the funtional type
        'functional': ['RPBE/SSSP_efficiency/gauss_smearing_0.1eV/dos_scf',
                       'PBE/SSSP_efficiency/cold_smearing_0.1eV/dos_scf',
                       ]
    }

    labels = {
        'relaxation': ['relaxed', 'scf'],
        'smearing': ['cold (0.1 eV)', 'cold (0.2 eV)', 'tetrahedron'],
        'functional': ['RPBE', 'PBE'],
    }

    # Loop over the different data to compare
    for data_type, compare in data_to_compare.items():
        # Figure for the projected density of states
        fig, ax = plt.subplots(len(METALS), len(METALS[1]), 
                               figsize=(13,12), sharey=True,
                               constrained_layout=True)
        figc, axc = plt.subplots(len(METALS), len(METALS[1]), 
                               figsize=(13,12), sharey=True,
                               constrained_layout=True)
        figo, axo = plt.subplots(len(METALS), len(METALS[1]), 
                               figsize=(13,12), sharey=True,
                               constrained_layout=True)
        for i, types in enumerate(compare):
            axc[0,2].plot(0, 0, color=COLORS[i], label=labels[data_type][i])
            axo[0,2].plot(0, 0, color=COLORS[i], label=labels[data_type][i])

        fig.suptitle(data_type)
        # Figure for the energies, if there is more than 
        # one dataset compare two separately and make 
        # those many more plots
        if len(compare) > 2:
            fige, axe = plt.subplots(1, len(compare), 
                                     constrained_layout=True)
        else:
            fige, axe = plt.subplots(1, 1, 
                                     constrained_layout=True)
        
        for i, case in enumerate(compare):
            # Read in the projected density of states
            label_pdos = case.replace('/', '_')
            with open(f'output/pdos_{label_pdos}.json') as handle:
                pdos_data = json.load(handle)

            # Keep track of which elements are computed
            used_ij = []

            # Color
            color = COLORS[i]
            
            for metal in pdos_data['slab']:
                # Do not plot metals in the remove list
                if metal in REMOVE_LIST:
                    continue

                # Get all the pdos
                energies, pdos, pdos_sp = pdos_data['slab'][metal]
                energies_c, pdos_c = pdos_data['C'][metal]
                energies_o, pdos_o = pdos_data['O'][metal]

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
                elif metal in METALS[3]:
                    i = 3
                    j = METALS[3].index(metal)
                else:
                    raise ValueError('Metal not in chosen list of metals.')
                if i == 0:
                    ax[i,j].plot(pdos_sp, energies, color=color, alpha=0.5, lw=3)
                else:
                    ax[i,j].plot(pdos, energies, color=color, alpha=0.5, lw=3)
                axc[i,j].plot(pdos_c, energies_c, color=color, alpha=0.5, lw=3)
                axo[i,j].plot(pdos_o, energies_o, color=color, alpha=0.5, lw=3)

                used_ij.append((i,j))

                ax[i,j].set_title(metal)
                ax[i,j].set_xticks([])
                ax[i,j].set_ylim(-12, 5)
                axo[i,j].set_title(metal)
                axo[i,j].set_xticks([])
                axo[i,j].set_ylim(-12, 5)
                axc[i,j].set_title(metal)
                axc[i,j].set_xticks([])
                axc[i,j].set_ylim(-12, 5)
                if j == 0:
                    ax[i,j].set_ylabel('$\epsilon - \epsilon_f$ (eV)')
                    axo[i,j].set_ylabel('$\epsilon - \epsilon_f$ (eV)')
                    axc[i,j].set_ylabel('$\epsilon - \epsilon_f$ (eV)')

        for i in [2, 3]: 
            ax[0,i].axis('off')
            axo[0,i].axis('off')
            axc[0,i].axis('off')
        ax[0,2].legend(loc='upper left')
        axc[0,2].legend(loc='upper left')
        axo[0,2].legend(loc='upper left')
        fig.savefig(f'output/pdos_compare_{data_type}.png')
        figc.savefig(f'output/pdos_c_compare_{data_type}.png')
        figo.savefig(f'output/pdos_o_compare_{data_type}.png')