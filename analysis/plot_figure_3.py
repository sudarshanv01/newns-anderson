"""Plot Figure 3 of the manuscript."""

import readline
from tkinter import CHORD
from venv import create
import numpy as np
import scipy
import json
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from scipy.optimize import curve_fit
from scipy import special, interpolate
import string
from fitting_functions import get_fitted_function
from create_coupling_elements import create_coupling_elements
from adjustText import adjust_text
from catchemi import ( NewnsAndersonLinearRepulsion,
                       NewnsAndersonNumerical,
                       NewnsAndersonDerivativeEpsd,
                       FitParametersNewnsAnderson
                     )  
from plot_figure_2 import normalise_na_quantities
import pickle
get_plot_params()
from scipy import stats
from plot_figure_4 import set_same_limits
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4

C_COLOR = 'tab:purple'
O_COLOR = 'tab:orange'

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 
ls_ads = ['--', '-']

def get_figure_layout():
    """Figure layout for the Figure 3 
    of the manuscript."""

    # Create gridspec for the figure
    fig = plt.figure(figsize=(6., 4.75))
    gs = plt.GridSpec(6, 3, figure=fig)

    # Create axes for the figure
    ax1 = [] ; ax2 = [] ; ax3 = []
    for i in range(2):
        ax1.append(fig.add_subplot(gs[3*i:3*(i+1),1]))
    for i in range(3):
        ax2.append(fig.add_subplot(gs[2*i:2*(i+1),2]))
    # Make the entire ax just for the density of states
    ax3 = fig.add_subplot(gs[:,0])
    ax3.set_ylabel(r'Projected density of states (arb. units)')
    ax3.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax3.set_xlim([-10,6])
    ax3.set_yticks([])
    ax3.axvline(x=0, color='tab:grey', linestyle='--')
    ax3.plot([], [], '-', color=O_COLOR, label='$p$-O*')
    ax3.plot([], [], '-', color=C_COLOR, label='$p$-C*')
    ax = np.array([ax1, ax2, ax3], dtype=object)

    for i in range(len(ADSORBATES)):
        ax[0][i].set_xlabel('DFT $\Delta E_{\mathrm{%s*}}$  (eV)'%ADSORBATES[i])
        ax[0][i].set_ylabel('$E_{\mathrm{chem}}$  %s* (eV)'%ADSORBATES[i])
        ax[1][i].set_ylabel('$E_{\mathrm{chem}}$  %s* (eV)'%ADSORBATES[i])
        ax[1][i].set_xlabel(r'$\epsilon_d$ (eV)')
    # ax[1][0].legend(loc='best')

    ax[1][2].set_xlabel(r'$E_{\mathrm{chem}}$ C* (eV)')
    ax[1][2].set_ylabel(r'$E_{\mathrm{chem}}$ O* (eV)')
    for i, color in enumerate(color_row):
        ax[0][1].plot([], [], color=color, label=f'{i+3}$d$')
    ax[0][1].legend(loc='best', fontsize=8)

    return fig, ax

if __name__ == '__main__':
    """Plot the model chemisorption energies against
    the DFT energies for C* and O* and also plot
    the chemisorption energies from the model to show
    that there is still a lack of scaling relations."""

    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    KEEP_LIST = []
    GRID_LEVEL = 'high' # 'high' or 'low'

    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = open('chosen_setup', 'r').read() 
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        c_parameters = json.load(f)
    adsorbate_params = {'O': o_parameters, 'C': c_parameters}
    # Read in the metal fitting splines
    with open(f"output/spline_objects.pkl", 'rb') as f:
        spline_objects = pickle.load(f)

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{COMP_SETUP[CHOSEN_SETUP]}.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    dft_Vsdsq = json.load(open(f"output/dft_Vsdsq.json"))
    no_of_bonds = yaml.safe_load(open('inputs/number_bonds.yaml', 'r'))
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    minmax_parameters = json.load(open('output/minmax_parameters.json'))

    # Load the pdos data
    pdos_data = json.load(open(f"output/pdos_{COMP_SETUP['dos']}.json"))

    # Parameters for the model
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1 ] # eV
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    EPS_VALUES = np.linspace(-30, 10, 1000)
    color_row = ['tab:red', 'tab:blue', 'tab:green',]
    if GRID_LEVEL == 'high':
        NUMBER_OF_METALS = 120
    elif GRID_LEVEL == 'low':
        NUMBER_OF_METALS = 10

    # Figure layout
    fig, ax = get_figure_layout()

    # Store the scaling energies for each adsorbate
    scaling_energies = defaultdict(lambda: defaultdict(list))
    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Plotting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
        alpha = adsorbate_params[adsorbate]['alpha']
        beta = adsorbate_params[adsorbate]['beta']
        constant_offest = adsorbate_params[adsorbate]['constant_offset']
        CONSTANT_DELTA0 = adsorbate_params[adsorbate]['delta0']
        final_params = [alpha, beta, constant_offest]

        # Store the parameters in order of metals in this list
        parameters = defaultdict(list)
        # Store the final DFT energies
        dft_energies = []
        metals = []

        for metal in data_from_energy_calculation[adsorbate]:

            if KEEP_LIST:
                if metal not in KEEP_LIST:
                    continue
            if REMOVE_LIST:
                if metal in REMOVE_LIST:
                    continue

            # get the parameters from DFT calculations
            width = data_from_dos_calculation[metal]['width']
            parameters['width'].append(width)
            d_band_centre = data_from_dos_calculation[metal]['d_band_centre']
            parameters['d_band_centre'].append(d_band_centre)

            # get the parameters from the energy calculations
            adsorption_energy = data_from_energy_calculation[adsorbate][metal]
            if isinstance(adsorption_energy, list):
                adsorption_energy = np.min(adsorption_energy)
            dft_energies.append(adsorption_energy)

            # get the idealised parameters 
            # Vsd = np.sqrt(data_from_LMTO['Vsdsq'][metal])
            Vsd = np.sqrt(dft_Vsdsq[metal])
            parameters['Vsd'].append(Vsd)

            # Get the metal filling
            filling = data_from_LMTO['filling'][metal]
            parameters['filling'].append(filling)

            # Store the order of the metals
            metals.append(metal)

            # Get the number of bonds based on the 
            # DFT calculation
            parameters['no_of_bonds'].append(no_of_bonds[CHOSEN_SETUP][metal])

        # Prepare the class for fitting routine 
        kwargs_fit = dict(
            eps_sp_min = EPS_SP_MIN,
            eps_sp_max = EPS_SP_MAX,
            eps = EPS_VALUES,
            Delta0_mag = CONSTANT_DELTA0,
            Vsd = parameters['Vsd'],
            width = parameters['width'],
            eps_a = eps_a,
            verbose = True,
            no_of_bonds = parameters['no_of_bonds'],
        )

        fitting_function =  FitParametersNewnsAnderson(**kwargs_fit)


        # Get the final hybridisation energy
        optimised_hyb = fitting_function.fit_parameters(final_params, parameters['d_band_centre'])

        # plot the parity line
        x = np.linspace(np.min(dft_energies)-0.3, np.max(dft_energies)+0.3, 2)
        ax[0][i].plot(x, x, '--', color='tab:grey', linewidth=1)

        # Fix the axes to the same scale 
        ax[0][i].set_xlim(np.min(x), np.max(x))
        ax[0][i].set_ylim(np.min(x), np.max(x))

        texts = []
        for j, metal in enumerate(metals):
            # Choose the colour based on the row of the TM
            if metal in FIRST_ROW:
                colour = color_row[0] 
            elif metal in SECOND_ROW:
                colour = color_row[1]
            elif metal in THIRD_ROW:
                colour = color_row[2]
            ax[0][i].plot(dft_energies[j], optimised_hyb[j], 'o', color=colour)
            texts.append(ax[0][i].text(dft_energies[j], optimised_hyb[j], metal, color=colour, alpha=0.5 ))
        # Check the fitting error
        slope, intercept, r_value, p_value, std_err = stats.linregress(dft_energies, optimised_hyb)
        print(f'{adsorbate} r2-value:', r_value**2)
        # Determine mean absolute error
        mean_absolute_error = np.mean(np.abs(optimised_hyb - dft_energies))
        print(f'{adsorbate} mean absolute error:', mean_absolute_error)
        print()


        # Also plot the chemisorption energies from the 
        # model against a continious variation of eps_d
        for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            parameters_metal = defaultdict(list)

            # get the metal fitting parameters
            s_fit = spline_objects[j]['s']
            Delta_anderson_fit = spline_objects[j]['Delta_anderson']
            wd_fit = spline_objects[j]['width']
            eps_d_fit = spline_objects[j]['eps_d']

            filling_min, filling_max = minmax_parameters[str(j)]['filling']
            eps_d_min, eps_d_max = minmax_parameters[str(j)]['eps_d']

            filling_range = np.linspace(filling_max, filling_min, NUMBER_OF_METALS)
            eps_d_range = np.linspace(eps_d_min, eps_d_max, NUMBER_OF_METALS)

            # First contruct a continuous variation of all parameters
            for k, filling in enumerate(filling_range):
                # Continuous setting of parameters for each 
                # continous variation of the metal
                width = wd_fit(filling) 
                eps_d = eps_d_fit(filling) 
                Vsdsq = create_coupling_elements(s_metal=s_fit(filling),
                                                s_Cu=s_data['Cu'],
                                                anderson_band_width=Delta_anderson_fit(filling),
                                                anderson_band_width_Cu=anderson_band_width_data['Cu'],
                                                r=s_fit(filling),
                                                r_Cu=s_data['Cu'],
                                                normalise_by_Cu=True,
                                                normalise_bond_length=True
                                                )
                Vsd = np.sqrt(Vsdsq)
                parameters_metal['Vsd'].append(Vsd)
                parameters_metal['eps_d'].append(eps_d)
                parameters_metal['width'].append(width)
                parameters_metal['filling'].append(filling)
                parameters_metal['no_of_bonds'] = parameters['no_of_bonds'][0]

            fitting_function.no_of_bonds = parameters_metal['no_of_bonds'] 
            fitting_function.Vsd = parameters_metal['Vsd']
            fitting_function.width = parameters_metal['width']

            # Chemisorption energies from the model plotted
            # against a continious variation of eps_d
            chemisorption_energy = fitting_function.fit_parameters( [alpha, beta, constant_offest], eps_d_range)
            ax[1][i].plot(eps_d_range, chemisorption_energy, color=color_row[j]) 
            # Store the chemisorption energies in scaling_energies
            # to plot a scaling line
            scaling_energies[j][adsorbate] = chemisorption_energy

        adjust_text(texts, ax=ax[0][i]) 
        ax[0][i].set_aspect('equal')
    
    # Plot the scaling line
    for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        ax[1][2].plot(scaling_energies[j]['C'], scaling_energies[j]['O'], color=color_row[j])

    # Add figure numbers
    alphabet = list(string.ascii_lowercase)
    ax[2].annotate(alphabet[0]+')', xy=(0.1, 0.9), xycoords='axes fraction')
    for i, a in enumerate(ax[0]):
        a.annotate(alphabet[i+1]+')', xy=(0.7, 0.1), xycoords='axes fraction')
    for j, a in enumerate(ax[1]):
        a.annotate(alphabet[i+j+2]+')', xy=(0.1, 0.5), xycoords='axes fraction')

    #-------- Plot the projected density of states --------#
    x_add = 0
    for row_index, row_metals in enumerate([FIRST_ROW,]): 
        for i, element in enumerate(row_metals):
            # Get the data for the element
            if element == 'X':
                continue
            if element in REMOVE_LIST:
                continue
            try:
                energies, pdos_metal_d, pdos_metal_sp = pdos_data['slab'][element]
                energies_C, pdos_C = pdos_data['C'][element]
                energies_O, pdos_O = pdos_data['O'][element]
            except KeyError:
                continue
            pdos_C = np.array(pdos_C)
            pdos_O = np.array(pdos_O)

            x_add += np.max(pdos_metal_d)
            pdos_C *= np.max(pdos_metal_d) / np.max(pdos_C)
            pdos_O *= np.max(pdos_metal_d) / np.max(pdos_O)
            pdos_metal_d = normalise_na_quantities(pdos_metal_d, x_add, per_max=False)
            pdos_metal_sp = normalise_na_quantities(pdos_metal_sp, x_add, per_max=False)
            pdos_C = normalise_na_quantities(pdos_C, x_add, per_max=False)
            pdos_O = normalise_na_quantities(pdos_O, x_add, per_max=False)
            # Set the maximum of the C, O pdos to the maximum of the metal pdos

            # Plot the pdos onto the metal states
            ax[2].plot(energies, pdos_metal_d, color='k',) 
            ax[2].fill_between(energies, x_add, pdos_metal_d, color='k',alpha=0.25) 
            ax[2].annotate(element, xy=(-8.5, pdos_metal_d[-1]+0.5), color='k')
            # Plot the C (sp) pdos
            ax[2].plot(energies_C, pdos_C, color=C_COLOR) 
            # Plot the O (sp) pdos
            ax[2].plot(energies_O, pdos_O, color=O_COLOR)
    
    # ax[2].legend(loc='best', fontsize=8)
    ax[2].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    fig.tight_layout()
    fig.savefig(f'output/figure_3_{COMP_SETUP[CHOSEN_SETUP]}.png', dpi=300)

