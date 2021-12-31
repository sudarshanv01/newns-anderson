"""Plot Figure 1 of the manuscript."""
from collections import defaultdict
import numpy as np
import json
from adjustText import adjust_text
from plot_params import get_plot_params
import string
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ase.visualize.plot import plot_atoms
from ase.io import read
get_plot_params()
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

C_COLOR = 'tab:grey'
O_COLOR = 'tab:red'

def get_plot_layout():
    """Create the plot layout for Figure 1."""
    fig = plt.figure(figsize=(12,10), constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.5, 1, 1])

    # Create the scaling plot first
    ax_scaling = fig.add_subplot(gs[0, 0:])
    ax_scaling.set_xlabel(r'$\Delta E_{\mathregular{C}}$ (eV)')
    ax_scaling.set_ylabel(r'$\Delta E_{\mathregular{O}}$ (eV)')
    ax_scaling.set_aspect('equal')

    # Create some axis for the images of structures
    ax_images = fig.add_subplot(gs[0, 0])
    ax_images.axis('off')

    # Create the plots with individual scaling
    ax_row = []
    for i in range(3):
        ax_row.append(fig.add_subplot(gs[1, i:1+i]))
        if i == 0:
            ax_row[i].set_ylabel(r'$\Delta E_{\mathregular{O}}$ (eV)')
        ax_row[i].set_xlabel(r'$\Delta E_{\mathregular{C}}$ (eV)')
    
    # Create plots with the d-band centre
    ax_dband = []
    for i in range(3):
        ax_dband.append(fig.add_subplot(gs[2, i:1+i]))
        if i == 0:
            ax_dband[i].set_ylabel(r'$\Delta E$ (eV)')
        ax_dband[i].set_xlabel(r'$\epsilon_u$ (eV)')
    
    ax_dband[0].plot([], [], 'o', color=C_COLOR, label='C*')
    ax_dband[0].plot([], [], 'o', color=O_COLOR, label='O*')

    ax_dband[0].legend(loc='best', fontsize=14)


    return fig, ax_scaling, ax_images, ax_row, ax_dband

def set_same_limits(axes, y_set=True, x_set=False):
    """Set the limits of all axes to the same value."""
    if y_set:
        ylims = []
        for i in range(len(axes)):
            ylims.append(axes[i].get_ylim())
        ylims = np.array(ylims).T
        # get the minimum and maximum 
        ymin = np.min(ylims[0])
        ymax = np.max(ylims[1])
        for ax in axes:
            ax.set_ylim([ymin, ymax])
    if x_set:
        xlims = []
        for i in range(len(axes)):
            xlims.append(axes[i].get_xlim())
        xlims = np.array(xlims).T
        # get the minimum and maximum
        xmin = np.min(xlims[0])
        xmax = np.max(xlims[1])
        for ax in axes:
            ax.set_xlim([xmin, xmax])


if __name__ == '__main__':
    """Plot the scaling relations from the energy file."""
    FUNCTIONAL = 'PBE_relax'
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']

    # Read the energy file.
    with open(f'output/adsorption_energies_{FUNCTIONAL}.json', 'r') as f:
        ads_energy = json.load(f)

    # Read the pdos file
    with open(f'output/pdos_moments_{FUNCTIONAL}.json', 'r') as f:
        pdos_data = json.load(f)

    # Plot separately for each row of the periodic table.
    fig, ax_s, ax_i, ax_r, ax_d = get_plot_layout()

    # Read in the atoms object
    atoms_all = read('output/all_structures_C.xyz', ':')
    plot_atoms(atoms_all[1], ax=ax_i)

    # Store the text output
    # texts = defaultdict(lambda: defaultdict(list))
    texts_s = []
    texts_r = defaultdict(list)
    texts_d = defaultdict(list)
    d_band_energies = []
    for metal in ads_energy['C']:
        if metal in REMOVE_LIST:
            continue
        if metal not in pdos_data:
            continue

        if metal in FIRST_ROW:
            color = 'k'
            index = FIRST_ROW.index(metal)
            row_i = 0
        elif metal in SECOND_ROW:
            color = 'k'
            index = SECOND_ROW.index(metal)
            row_i = 1
        elif metal in THIRD_ROW:
            color = 'k'
            index = THIRD_ROW.index(metal)
            row_i = 2
        
        DeltaE_C = ads_energy['C'][metal]
        DeltaE_O = ads_energy['O'][metal]
        d_band_desc = pdos_data[metal]['d_band_upper_edge']
        d_band_centre = pdos_data[metal]['d_band_centre']

        # Plot all the DeltaE_C and DeltaE_O
        # ax_s.plot(DeltaE_C, DeltaE_O, 'o', color=color)
        d_band_energies.append([DeltaE_C, DeltaE_O, d_band_centre])
        texts_s.append( ax_s.text(DeltaE_C, DeltaE_O, metal, color=color, fontsize=14))

        # Plot the individual energies against themselves or against
        ax_r[row_i].plot(DeltaE_C, DeltaE_O, 'o', color=color)
        texts_r[row_i].append( ax_r[row_i].text(DeltaE_C, DeltaE_O, metal, color=color, fontsize=14))

        # Plot against the d-band centre
        ax_d[row_i].plot(d_band_desc, DeltaE_O, 'o', color=O_COLOR)
        texts_d[row_i].append( ax_d[row_i].text(d_band_desc, DeltaE_O, metal, color=O_COLOR, fontsize=14))
        ax_d[row_i].plot(d_band_desc, DeltaE_C, 'o', color=C_COLOR)
        texts_d[row_i].append( ax_d[row_i].text(d_band_desc, DeltaE_C, metal, color=C_COLOR, fontsize=14))

    all_C, all_O, all_dband = np.array(d_band_energies).T 
    cax = ax_s.scatter(all_C, all_O,  cmap='coolwarm', c=all_dband, marker='o')
    # Plot colorbar
    fig.colorbar(cax, ax=ax_s, label='$\epsilon_d$ (eV)')

    adjust_text(texts_s, ax=ax_s)
    for row_i, row in texts_r.items(): 
        adjust_text(row, ax=ax_r[row_i])
    for row_i, row in texts_d.items(): 
        adjust_text(row, ax=ax_d[row_i])
    
    for i in range(3):
        ax_r[i].annotate(f'{i+3}d', xy=(0.05, 0.9), xycoords='axes fraction', color='tab:blue') 


    all_axes = [ax_i, ax_s, *ax_r, *ax_d]
    # Add figure numbers
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(all_axes):
        a.annotate(alphabet[i]+')', xy=(0.05, 1.05), xycoords='axes fraction') 

    set_same_limits(ax_r, y_set=True)
    set_same_limits(ax_d, y_set=True)

    fig.savefig(f'output/figure_1_{FUNCTIONAL}.png', dpi=300)