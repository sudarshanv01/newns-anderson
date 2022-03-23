"""Plot Figure 1 of the manuscript."""
from collections import defaultdict
import numpy as np
import json
from adjustText import adjust_text
from plot_params import get_plot_params
import string
import yaml
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ase.visualize.plot import plot_atoms
from ase.io import read
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4
get_plot_params()
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

C_COLOR = 'tab:blue'
O_COLOR = 'tab:red'
COLORS_ROW = ['tab:red', 'tab:blue', 'tab:green']

def get_plot_layout():
    """Create the plot layout for Figure 1."""
    fig = plt.figure(figsize=(5.25, 4), constrained_layout=True)
    gs = gridspec.GridSpec(6, 6, figure=fig)

    # Create the scaling plot first
    ax_scaling = fig.add_subplot(gs[0:4, 0:4])
    ax_scaling.set_xlabel(r'$\Delta E_{\mathregular{C}}$ (eV)')
    ax_scaling.set_ylabel(r'$\Delta E_{\mathregular{O}}$ (eV)')
    # ax_scaling.set_aspect('equal')

    # Create the plots with individual scaling
    ax_row = []
    for i in range(3):
        ax_row.append(fig.add_subplot(gs[2*i:2*(i+1), 4:]))
        # if i == 0:
        ax_row[i].set_ylabel(r'$\Delta E_{\mathregular{O}}$ (eV)')
        ax_row[i].set_xlabel(r'$\Delta E_{\mathregular{C}}$ (eV)')
    
    # Create plots with the d-band centre
    ax_dband = []
    for i in range(2):
        ax_dband.append(fig.add_subplot(gs[4:, 2*i:2*(i+1)]))
        ax_dband[i].set_ylabel(r'$\Delta E_{\mathrm{%s}}$ (eV)'%ADSORBATES[i])
        ax_dband[i].set_xlabel(r'$\epsilon_d$ (eV)')

    return fig, ax_scaling, ax_dband, ax_row

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
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = open('chosen_setup', 'r').read() 
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']

    # Read the energy file.
    with open(f"output/adsorption_energies_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        ads_energy = json.load(f)

    # Read the pdos file
    with open(f"output/pdos_moments_{COMP_SETUP['dos']}.json", 'r') as f:
        pdos_data = json.load(f)
    
    ADSORBATES = ['O', 'C']

    # Plot separately for each row of the periodic table.
    fig, ax_s, ax_d, ax_r = get_plot_layout()


    # Store the text output
    # texts = defaultdict(lambda: defaultdict(list))
    texts_s = []
    texts_r = defaultdict(list)
    texts_d = defaultdict(list)
    d_band_energies = []
    energies_row = defaultdict(list)
    for metal in ads_energy['C']:
        if metal in REMOVE_LIST:
            continue
        if metal not in pdos_data:
            continue

        if metal in FIRST_ROW:
            color = 'tab:red'
            index = FIRST_ROW.index(metal)
            row_i = 0
        elif metal in SECOND_ROW:
            color = 'tab:blue'
            index = SECOND_ROW.index(metal)
            row_i = 1
        elif metal in THIRD_ROW:
            color = 'tab:green'
            index = THIRD_ROW.index(metal)
            row_i = 2
        
        DeltaE_C = ads_energy['C'][metal]
        DeltaE_O = ads_energy['O'][metal]

        if isinstance(DeltaE_O, list):
            DeltaE_O = np.min(DeltaE_O)
        if isinstance(DeltaE_C, list):
            DeltaE_C = np.min(DeltaE_C)

        d_band_desc = pdos_data[metal]['d_band_centre']
        d_band_centre = pdos_data[metal]['d_band_centre']

        # Plot all the DeltaE_C and DeltaE_O
        d_band_energies.append([DeltaE_C, DeltaE_O, d_band_centre])
        texts_s.append( ax_s.text(DeltaE_C, DeltaE_O, metal, color=color, fontsize=7, alpha=0.5))

        # Plot the individual energies against themselves or against
        ax_r[row_i].plot(DeltaE_C, DeltaE_O, 'o', color=color)
        texts_r[row_i].append( ax_r[row_i].text(DeltaE_C, DeltaE_O, metal, color=color, fontsize=7, alpha=0.5 ))
        ax_s.plot(DeltaE_C, DeltaE_O, 'o', color=color)
        energies_row[row_i].append([DeltaE_C, DeltaE_O, d_band_desc])

        # Plot against the d-band centre
        ax_d[0].plot(d_band_desc, DeltaE_O, 'o', color=color)
        texts_d[0].append( ax_d[0].text(d_band_desc, DeltaE_O, metal, color=color, fontsize=5, alpha=0.5))
        ax_d[1].plot(d_band_desc, DeltaE_C, 'o', color=color)
        texts_d[1].append( ax_d[1].text(d_band_desc, DeltaE_C, metal, color=color, fontsize=5, alpha=0.5))

        # If the metal is Pt make a point on the d-band centre
        if metal == 'Pt':
            Pt_E_C, Pt_E_O = DeltaE_C, DeltaE_O

    all_C, all_O, all_dband = np.array(d_band_energies).T 
    # cax = ax_s.scatter(all_C, all_O,  cmap='cividis', c=all_dband, marker='o')

    # Fit all the energies to a straight line
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_C, all_O)
    print('Overall r2-value:', r_value**2)
    p_fit = np.poly1d([slope, intercept])
    minmax_allC = [np.min(all_C), np.max(all_C)]
    # ax_s.plot(minmax_allC, p_fit(minmax_allC), '--', color='tab:grey', alpha=0.5)
    for row in range(3):
        row_ener_C, row_ener_O, row_dband = np.array(energies_row[row]).T
        # Sort all quantities with respect to the d-band centre
        row_ener_C, row_ener_O, row_dband = row_ener_C[row_dband.argsort()], row_ener_O[row_dband.argsort()], row_dband[row_dband.argsort()]
        # Remove the last value of all quantities
        if row == 0:
            cutoff = 2
        else:
            cutoff = 1
        row_ener_C = row_ener_C[cutoff:]
        row_ener_O = row_ener_O[cutoff:]
        row_dband = row_dband[cutoff:]
        slope, intercept, r_value, p_value, std_err = stats.linregress(row_ener_C, row_ener_O)
        print(f'Row {row} r2-value:', r_value**2)
        p_fit = np.poly1d([slope, intercept])
        minmax_rowC = [np.min(row_ener_C), np.max(row_ener_C)+0.05]
        # ax_r[row].plot(row_ener_C, row_ener_O, 'o', color=COLORS_ROW[row])
        # ax_r[row].plot(minmax_rowC, p_fit(minmax_rowC), '--', color=COLORS_ROW[row], alpha=0.5)

    # Plot colorbar
    # cbaxes = inset_axes(ax_s, width="30%", height="3%", loc=2) 
    # cbar = fig.colorbar(cax, cax=cbaxes, label='$\epsilon_d$ (eV)', orientation='horizontal') 

    adjust_text(texts_s, ax=ax_s)
    for row_i, row in texts_r.items(): 
        adjust_text(row, ax=ax_r[row_i])
    adjust_text(texts_d[0], ax=ax_d[0])
    adjust_text(texts_d[1], ax=ax_d[1])

    
    for i in range(3):
        ax_r[i].annotate(f'{i+3}$d$', xy=(0.8, 0.2), xycoords='axes fraction', color=COLORS_ROW[i]) 


    all_axes = [ax_s,  *ax_d, *ax_r]
    # Add figure numbers
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(all_axes):
        if i in [1, 2]:
            xy=(0.05, 0.1)
        elif i in [0]:
            xy=(0.8, 0.1)
        else:
            xy=(0.05, 0.8)
        a.annotate(alphabet[i]+')', xy=xy, xycoords='axes fraction') 

    # set_same_limits(ax_r, y_set=True)
    # set_same_limits(ax_d, y_set=True)

    fig.savefig(f'output/figure_1_{COMP_SETUP[CHOSEN_SETUP]}.png', dpi=300)