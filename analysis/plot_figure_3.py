"""Plot Figure 3 of the manuscript."""

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
from adjustText import adjust_text
from catchemi import ( NewnsAndersonLinearRepulsion,
                       NewnsAndersonNumerical,
                       NewnsAndersonDerivativeEpsd,
                       FitParametersNewnsAnderson
                     )  
get_plot_params()
from scipy import stats
from plot_figure_4 import set_same_limits
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 
ls_ads = ['--', '-']

def get_figure_layout():
    """Figure layout for the Figure 3 
    of the manuscript."""

    # Create gridspec for the figure
    fig = plt.figure(figsize=(5.065, 4.75), constrained_layout=True)
    gs = plt.GridSpec(6, 2, figure=fig)

    # Create axes for the figure
    ax1 = [] ; ax2 = [] ; ax3 = []
    for i in range(2):
        ax1.append(fig.add_subplot(gs[3*i:3*(i+1),0]))
    for i in range(3):
        ax2.append(fig.add_subplot(gs[2*i:2*(i+1),1]))
    ax = np.array([ax1, ax2], dtype=object)

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
    GRID_LEVEL = 'low' # 'high'

    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters.json", 'r') as f:
        c_parameters = json.load(f)
    with open(f"output/fitting_metal_parameters.json", 'r') as f:
        metal_parameters = json.load(f)

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{COMP_SETUP['energy']}.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    dft_Vsdsq = json.load(open(f"output/dft_Vsdsq.json"))

    # Parameters for the model
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1 ] # eV
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CONSTANT_DELTA0 = 0.1
    EPS_VALUES = np.linspace(-20, 20, 1000)
    color_row = ['tab:red', 'tab:blue', 'tab:green',]
    if GRID_LEVEL == 'high':
        NUMBER_OF_METALS = 120
    elif GRID_LEVEL == 'low':
        NUMBER_OF_METALS = 10

    # Functions of Vsd and width as a function of the filling
    function_Vsd, function_Vsd_p = get_fitted_function('Vsd') 
    function_wd, function_wd_p = get_fitted_function('wd')

    # Figure layout
    fig, ax = get_figure_layout()

    # Store the scaling energies for each adsorbate
    scaling_energies = defaultdict(lambda: defaultdict(list))
    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Plotting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
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
            dft_energies.append(adsorption_energy)

            # get the idealised parameters 
            # Vsd = np.sqrt(data_from_LMTO['Vsdsq'][metal])
            Vsd = np.sqrt(dft_Vsdsq[adsorbate][metal])
            parameters['Vsd'].append(Vsd)

            # Get the metal filling
            filling = data_from_LMTO['filling'][metal]
            parameters['filling'].append(filling)

            # Store the order of the metals
            metals.append(metal)

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
        )

        fitting_function =  FitParametersNewnsAnderson(**kwargs_fit)

        previous_calc = json.load(open(f'output/{adsorbate}_parameters.json'))
        alpha = previous_calc['alpha']
        beta = previous_calc['beta']
        constant_offest = previous_calc['constant_offset']
        final_params = [alpha, beta, constant_offest]

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
            texts.append(ax[0][i].text(dft_energies[j], optimised_hyb[j], metal, color=colour, ))
        # Check the fitting error
        slope, intercept, r_value, p_value, std_err = stats.linregress(dft_energies, optimised_hyb)
        print(f'{adsorbate} r2-value:', r_value**2)
        # Determine mean absolute error
        mean_absolute_error = np.mean(np.abs(optimised_hyb - dft_energies))
        print(f'{adsorbate} mean absolute error:', mean_absolute_error)
        print()

        adjust_text(texts, ax=ax[0][i]) 
        ax[0][i].set_aspect('equal')

        # Also plot the chemisorption energies from the 
        # model against a continious variation of eps_d
        for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            parameters_metal = defaultdict(list)

            # get the metal fitting parameters
            Vsd_fit = metal_parameters['Vsd'][str(j)]
            wd_fit = metal_parameters['width'][str(j)]
            epsd_filling_fit = metal_parameters['epsd_filling'][str(j)]
            filling_min, filling_max = metal_parameters['filling_minmax'][str(j)]
            eps_d_min, eps_d_max = metal_parameters['eps_d_minmax'][str(j)]
            filling_range = np.linspace(filling_max, filling_min, NUMBER_OF_METALS)
            eps_d_range = np.linspace(eps_d_min, eps_d_max, NUMBER_OF_METALS)

            # First contruct a continuous variation of all parameters
            for k, filling in enumerate(filling_range):
                # Continuous setting of parameters for each 
                # continous variation of the metal
                Vsd = function_Vsd(filling, *Vsd_fit )
                width = function_wd(filling, *wd_fit )
                eps_d = eps_d_range[k]                

                parameters_metal['Vsd'].append(Vsd)
                parameters_metal['eps_d'].append(eps_d)
                parameters_metal['width'].append(width)
                parameters_metal['filling'].append(filling)

            fitting_function.Vsd = parameters_metal['Vsd']
            fitting_function.width = parameters_metal['width']

            # Chemisorption energies from the model plotted
            # against a continious variation of eps_d
            chemisorption_energy = fitting_function.fit_parameters( [alpha, beta, constant_offest], eps_d_range)
            ax[1][i].plot(eps_d_range, chemisorption_energy, color=color_row[j]) 
            # Store the chemisorption energies in scaling_energies
            # to plot a scaling line
            scaling_energies[j][adsorbate] = chemisorption_energy
    
    # Plot the scaling line
    for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        ax[1][2].plot(scaling_energies[j]['C'], scaling_energies[j]['O'], color=color_row[j])

    # Add figure numbers
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(ax[0]):
        a.annotate(alphabet[i]+')', xy=(0.1, 0.5), xycoords='axes fraction')
    for j, a in enumerate(ax[1]):
        a.annotate(alphabet[i+j+1]+')', xy=(0.1, 0.5), xycoords='axes fraction')
    
    # Set the same axes limits for each row
    # for i, row in enumerate(ax):
    #     if i == 2:
    #         continue
    #     set_same_limits(row, y_set=True, x_set=False)

    fig.savefig(f'output/figure_3.png', dpi=300)

