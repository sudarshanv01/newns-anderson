"""Get fits of the metal parameters with the filling fraction."""
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import yaml
import marshal
from fitting_functions import ( func_a_by_r, func_exp, func_a_r_sq, 
                                get_fit_for_Vsd, get_fit_for_wd,
                                get_fit_for_epsd, function_linear, 
                                function_quadratic )
from plot_params import get_plot_params
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 


if __name__ == '__main__':
    """Plot the fitting parameters and decide the range to be used."""

    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']

    # Plot the fits
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    ax[0].set_xlabel('Filling fraction')
    ax[0].set_ylabel('$V_{sd}$ (eV)')
    ax[1].set_xlabel('Filling fraction')
    ax[1].set_ylabel('$w_{d}$ (eV)')
    ax[2].set_ylabel('Filling fraction')
    ax[2].set_xlabel('$\epsilon_d$ (eV)')

    # Plot the fits with the d-band centre
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax2[0].set_xlabel('$\epsilon_d$ (eV)')
    ax2[0].set_ylabel('$V_{sd}$')
    ax2[1].set_xlabel('$\epsilon_d$ (eV)')
    ax2[1].set_ylabel('$w_{d}$')

    # Input parameters to help with the dos from Newns-Anderson
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{COMP_SETUP['energy']}.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Plot the fitting of Vsd and weights with filling fractions
    parameters = defaultdict(lambda: defaultdict(list))

    for i, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        for j, metal in enumerate(metal_row):
            try:
                width = data_from_dos_calculation[metal]['width']
                filling = data_from_LMTO['filling'][metal]
                Vsdsq = data_from_LMTO['Vsdsq'][metal]
            except KeyError:
                continue

            parameters['Vsd'][i].append(np.sqrt(Vsdsq))
            parameters['filling'][i].append(filling)
            parameters['width'][i].append(width)
            eps_d = data_from_dos_calculation[metal]['d_band_centre']
            parameters['eps_d'][i].append(eps_d)
            if metal in REMOVE_LIST:
                continue
            parameters['eps_d_rel'][i].append(eps_d)
            parameters['filling_rel'][i].append(filling)
            parameters['metal'][i].append(metal)
    
    # Get the fits for the metal rows
    fitting_parameters = defaultdict(lambda: defaultdict(list))
    filling_range = np.linspace(0.1, 1.1, 50)

    # Cycle of colors
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i in range(3):
        # First fit everything with respect to the filling fraction
        fit_Vsd = get_fit_for_Vsd(parameters['filling'][i], parameters['Vsd'][i])
        fit_width = get_fit_for_wd(parameters['filling'][i], parameters['width'][i])
        fit_epsd_filling = get_fit_for_epsd(parameters['eps_d'][i], parameters['filling'][i])

        # get the eps_d range and the filling range
        eps_drel_minmax = [ np.min(parameters['eps_d_rel'][i]), np.max(parameters['eps_d_rel'][i]) ]
        eps_d_minmax = [ np.min(parameters['eps_d'][i]), np.max(parameters['eps_d'][i]) ]
        filling_minmax = [ np.min(parameters['filling_rel'][i]), np.max(parameters['filling_rel'][i]) ]

        # Plotting based on the epsd range of the metal row
        epsd_range = np.linspace(eps_d_minmax[0], eps_d_minmax[1], 50)

        # Get the same fits with the d-band centre
        kwargs = {'input_epsd':True, 'fitted_epsd_to_filling':fit_epsd_filling[0]}
        
        fitting_parameters['Vsd'][i] = fit_Vsd[0]
        fitting_parameters['width'][i] = fit_width[0]
        fitting_parameters['epsd_filling'][i] = fit_epsd_filling[0]

        ax[0].plot(parameters['filling'][i], parameters['Vsd'][i], 'o', label=f'{i+1}-row', color=colors[i])
        ax[0].plot(filling_range, func_a_by_r(filling_range, *fit_Vsd[0]), '--', color=colors[i])

        ax[1].plot(parameters['filling'][i], parameters['width'][i], 'o', label=f'{i+1}-row', color=colors[i])
        ax[1].plot(filling_range, func_a_r_sq(filling_range, *fit_width[0]), '--', color=colors[i])

        ax[2].plot(parameters['eps_d'][i], parameters['filling'][i], 'o', label=f'{i+1}-row', color=colors[i])
        ax[2].plot(epsd_range, function_linear(epsd_range, *fit_epsd_filling[0]), '--', color=colors[i])

        ax2[0].plot(parameters['eps_d'][i], parameters['Vsd'][i], 'o', label=f'{i+1}-row', color=colors[i])
        ax2[0].plot(epsd_range, func_a_by_r(epsd_range, *fit_Vsd[0], **kwargs), '--', color=colors[i])

        ax2[1].plot(parameters['eps_d'][i], parameters['width'][i], 'o', label=f'{i+1}-row', color=colors[i])
        ax2[1].plot(epsd_range, func_a_r_sq(epsd_range, *fit_width[0], **kwargs), '--', color=colors[i])


        fitting_parameters['eps_d_minmax'][i] = eps_drel_minmax
        fitting_parameters['filling_minmax'][i] = filling_minmax
    
    fig.savefig(f'output/fitting_metal_parameters.png')
    fig2.savefig(f'output/fitting_metal_parameters_dband.png')

    # Write out the fitting parameters to a json file
    json.dump(fitting_parameters, open(f'output/fitting_metal_parameters.json', 'w'), indent=4)

