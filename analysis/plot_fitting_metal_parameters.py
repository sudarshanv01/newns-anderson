"""Get fits of the metal parameters with the filling fraction."""
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import yaml
import marshal
from fitting_functions import ( interpolate_quantity,
                                get_fitted_function )
from plot_params import get_plot_params
from ase import units
import pickle
from create_coupling_elements import create_coupling_elements
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 


if __name__ == '__main__':
    """Plot the fitting parameters and decide the range to be used."""

    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']

    # Plot the fits
    fig, axa = plt.subplots(1, 5, figsize=(6.75, 2), constrained_layout=True, squeeze=False)

    ax = axa[0,:]
    ax[0].set_xlabel('Filling fraction')
    ax[0].set_ylabel('s (Bohr)')
    ax[1].set_xlabel('Filling fraction')
    ax[1].set_ylabel('Anderson $\Delta$ (a.u.)')
    ax[2].set_xlabel('Filling fraction')
    ax[2].set_ylabel('$V_{sd}^2$ (eV)')
    ax[3].set_xlabel('Filling fraction')
    ax[3].set_ylabel('$w_{d}$ (eV)')
    ax[4].set_xlabel('Filling fraction')
    ax[4].set_ylabel('$\epsilon_d$ (eV)')

    # Input parameters to help with the dos from Newns-Anderson
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = 'energy'
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{COMP_SETUP[CHOSEN_SETUP]}.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    dft_Vsdsq = json.load(open('output/dft_Vsdsq.json'))

    # Get the bond length-corrected Vsd 
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    s_data = data_from_LMTO['s']
    Vsdsq_data = data_from_LMTO['Vsdsq']
    no_of_bonds = yaml.safe_load(open('inputs/number_bonds.yaml', 'r'))

    # Cycle of colors
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    # Get the fits for the metal rows
    fitting_parameters = defaultdict( lambda: defaultdict(lambda: defaultdict(list)))

    # Get the fitting functions (without the derivatives)
    fitted_function_Md, fitted_function_bl = get_fitted_function(quantity_y='Vsd', return_derivative=False)
    fitted_function_wd = get_fitted_function(quantity_y='wd', return_derivative=False)
    
    # Plot the fitting of Vsd and weights with filling fractions
    parameters = defaultdict(lambda: defaultdict(list))
    # Plot everything on this axis
    ax = axa[0,:]

    for i, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        for j, metal in enumerate(metal_row):
            if metal in REMOVE_LIST:
                continue
            try:
                width = data_from_dos_calculation[metal]['width']
                filling = data_from_LMTO['filling'][metal]
                Vsdsq = dft_Vsdsq[metal]
            except KeyError:
                continue

            parameters['Vsdsq'][i].append(Vsdsq)
            parameters['filling'][i].append(filling)
            parameters['width'][i].append(width)
            eps_d = data_from_dos_calculation[metal]['d_band_centre']
            parameters['eps_d'][i].append(eps_d)
            if metal in REMOVE_LIST:
                continue
            parameters['eps_d_rel'][i].append(eps_d)
            parameters['filling_rel'][i].append(filling)
            parameters['metal'][i].append(metal)
            parameters['s'][i].append(s_data[metal])
            parameters['Delta_anderson'][i].append(anderson_band_width_data[metal])


    # Iterate through each row of transition metals
    spline_objects = defaultdict(dict)
    minmax_dict = defaultdict(lambda: defaultdict(list))
    for i in range(3):
        
        # get the eps_d range and the filling range
        eps_drel_minmax = [ np.min(parameters['eps_d_rel'][i]), np.max(parameters['eps_d_rel'][i]) ]
        eps_d_minmax = [ np.min(parameters['eps_d'][i]), np.max(parameters['eps_d'][i]) ]
        filling_minmax = [ np.min(parameters['filling_rel'][i]), np.max(parameters['filling_rel'][i]) ]
        # minmax_dict[i]['eps_d_rel'] = eps_drel_minmax
        minmax_dict[i]['eps_d'] = eps_d_minmax
        minmax_dict[i]['filling'] = filling_minmax

        # Plotting based on the epsd range of the metal row
        epsd_range = np.linspace(eps_d_minmax[0], eps_d_minmax[1], 50)
        filling_range = np.linspace(filling_minmax[0], filling_minmax[1], 50)

        # Interpolate the fitting parameters
        spline_epsd = interpolate_quantity(parameters['filling'][i], parameters['eps_d'][i])
        spline_s = interpolate_quantity(parameters['filling'][i], parameters['s'][i])
        spline_Delta_anderson = interpolate_quantity(parameters['filling'][i], parameters['Delta_anderson'][i])
        spline_width = interpolate_quantity(parameters['filling'][i], parameters['width'][i])
        spline_filling = np.poly1d(np.polyfit(parameters['eps_d'][i], parameters['filling'][i], 1)) 
        # interp1d(parameters['eps_d'][i], parameters['filling'][i])
        spline_Vsdsq = interpolate_quantity(parameters['filling'][i], parameters['Vsdsq'][i])
        spline_Vsd = interpolate_quantity(parameters['filling'][i], np.sqrt(parameters['Vsdsq'][i]))
        spline_objects[i]['eps_d'] = spline_epsd
        spline_objects[i]['s'] = spline_s
        spline_objects[i]['Delta_anderson'] = spline_Delta_anderson
        spline_objects[i]['width'] = spline_width
        spline_objects[i]['filling'] = spline_filling
        spline_objects[i]['Vsdsq'] = spline_Vsdsq
        spline_objects[i]['Vsd'] = spline_Vsd

        # Get the coupling elements based on the fit
        Vsdsq_interp = []
        for filling in filling_range:
            s_filling = spline_s(filling)
            Delta_anderson_filling = spline_Delta_anderson(filling)
            Vsdsq_filling = s_filling**-3 * Delta_anderson_filling

            # Reference to Cu
            Vsdsq_filling /= s_data['Cu']**-3 * anderson_band_width_data['Cu']
            Vsdsq_interp.append(Vsdsq_filling)

        # Plot the real and interpolated values of s, Delta_anderson, Vsdsq, width and eps_d
        ax[0].plot(parameters['filling'][i], parameters['s'][i], 'o', color=colors[i], label=f'{metal_row[i]}')
        ax[0].plot(filling_range, spline_s(filling_range), color=colors[i], ls='-')
        ax[1].plot(parameters['filling'][i], parameters['Delta_anderson'][i], 'o', color=colors[i], label=f'{metal_row[i]}')
        ax[1].plot(filling_range, spline_Delta_anderson(filling_range), color=colors[i], ls='-')
        ax[2].plot(parameters['filling'][i], parameters['Vsdsq'][i], 'o', color=colors[i], label=f'{metal_row[i]}')
        # ax[2].plot(filling_range, Vsdsq_interp, color=colors[i], label=f'{metal_row[i]}')
        ax[2].plot(filling_range, spline_Vsdsq(filling_range), color=colors[i], ls='-')
        ax[3].plot(parameters['filling'][i], parameters['width'][i], 'o', color=colors[i], label=f'{metal_row[i]}')
        ax[3].plot(filling_range, spline_width(filling_range), color=colors[i], ls='-')
        ax[4].plot(parameters['filling'][i], parameters['eps_d'][i], 'o', color=colors[i], label=f'{metal_row[i]}')
        # ax[4].plot(filling_range, spline_epsd(filling_range), color=colors[i], ls='-')
        ax[4].plot(spline_filling(epsd_range), epsd_range, color=colors[i], ls='-')

    # Write out spline objects as a pickle file
    with open('output/spline_objects.pkl', 'wb') as f:
        pickle.dump(spline_objects, f)
    with open('output/minmax_parameters.json', 'w') as handle:
        json.dump(minmax_dict, handle, indent=4)

    fig.savefig(f'output/fitting_metal_parameters.png', dpi=300)
