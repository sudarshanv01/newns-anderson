"""Get fits of the metal parameters with the filling fraction."""
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from plot_params import get_plot_params
import yaml
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

def func_a_by_r(x, a):
    return a / x

def func_a_r_sq(x, a, b):
    return b - a * ( x - 0.7) **2

def get_fit_for_Vsdsq(x, y):
    """Get fit and error for the square of the Vsd."""
    popt, pcov = curve_fit(func_a_by_r, x, y)
    return list(popt), list(pcov)

def get_fit_for_wd(x, y):
    """Get the fit and error for the width.""" 
    popt, pcov = curve_fit(func_a_r_sq, x, y)
    return list(popt), list(pcov)
    # return [ np.mean(y) ], [ np.std(x) ]

if __name__ == '__main__':
    """Plot the fitting parameters and decide the range to be used."""

    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']

    # Plot the fits
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].set_xlabel('Filling fraction')
    ax[0].set_ylabel('$V_{sd}^2$')
    ax[1].set_xlabel('Filling fraction')
    ax[1].set_ylabel('$w_{d}$')


    # Input parameters to help with the dos from Newns-Anderson
    FUNCTIONAL = 'PBE_scf'
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{FUNCTIONAL}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{FUNCTIONAL}.json'))
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

            parameters['Vsdsq'][i].append(Vsdsq)
            parameters['filling'][i].append(filling)
            parameters['width'][i].append(width)
            if metal in REMOVE_LIST:
                continue
            parameters['filling_rel'][i].append(filling)
            eps_d = data_from_dos_calculation[metal]['d_band_centre']
            parameters['eps_d'][i].append(eps_d)
    
    # Get the fits for the metal rows
    fitting_parameters = defaultdict(lambda: defaultdict(list))
    filling_range = np.linspace(0.1, 1.1, 50)

    # Cycle of colors
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for i in range(3):
        fit_Vsdsq = get_fit_for_Vsdsq(parameters['filling'][i], parameters['Vsdsq'][i])
        fit_width = get_fit_for_wd(parameters['filling'][i], parameters['width'][i])

        fitting_parameters['Vsdsq'][i] = fit_Vsdsq[0]
        fitting_parameters['width'][i] = fit_width[0]

        ax[0].plot(parameters['filling'][i], parameters['Vsdsq'][i], 'o', label=f'{i+1}-row', color=colors[i])
        ax[0].plot(filling_range, func_a_by_r(filling_range, *fit_Vsdsq[0]), '--', color=colors[i])

        ax[1].plot(parameters['filling'][i], parameters['width'][i], 'o', label=f'{i+1}-row', color=colors[i])
        ax[1].plot(filling_range, func_a_r_sq(filling_range, *fit_width[0]), '--', color=colors[i])
        # ax[1].axhline(fit_width[0], color=colors[i], linestyle='--')

        # get the eps_d range and the filling range
        eps_d_minmax = [ np.min(parameters['eps_d'][i]), np.max(parameters['eps_d'][i]) ]
        filling_minmax = [ np.min(parameters['filling_rel'][i]), np.max(parameters['filling_rel'][i]) ]

        fitting_parameters['eps_d_minmax'][i] = eps_d_minmax
        fitting_parameters['filling_minmax'][i] = filling_minmax
    
    fig.savefig(f'output/fitting_metal_parameters_{FUNCTIONAL}.png')

    # Write out the fitting parameters to a json file
    json.dump(fitting_parameters, open(f'output/fitting_metal_parameters_{FUNCTIONAL}.json', 'w'), indent=4)
