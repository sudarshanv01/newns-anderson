"""Get parameters for the Newns-Anderson model and plot Figure 3 of the manuscript."""
import sys
import json
import yaml
import numpy as np
from dataclasses import dataclass
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical, NorskovNewnsAnderson 
from collections import defaultdict
from scipy.optimize import minimize, least_squares, leastsq, curve_fit
from scipy import odr
from pprint import pprint
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from adjustText import adjust_text
from yaml import safe_load
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

if __name__ == '__main__':
    """Determine the fitting parameters for a particular adsorbate."""
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    KEEP_LIST = []

    if len(sys.argv) > 1:
        if sys.argv[1] == 'restart':
            restart = True
        else:
            restart = False 
    else:
        restart = False

    # Choose a sequence of adsorbates
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1 ] # eV
    EPS_VALUES = np.linspace(-20, 20, 1000)
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CONSTANT_DELTA0 = 0.1
    print(f"Fitting parameters for adsorbate {ADSORBATES} with eps_a {EPS_A_VALUES}")

    # The functional and type of calculation we will use
    # scf only calculations in order to avoid any noise and look only for 
    # the electronic structure contribution
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))['group'][0]

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{COMP_SETUP}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{COMP_SETUP}.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Plot the fitted and the real adsorption energies
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.), constrained_layout=True)
    for i in range(len(ax)):
        ax[i].set_xlabel('DFT energy (eV)')
        ax[i].set_ylabel('Chemisorption energy (eV)')
        ax[i].set_title(f'{ADSORBATES[i]}* with $\epsilon_a=$ {EPS_A_VALUES[i]} eV')

    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Fitting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
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
            Vsd = np.sqrt(data_from_LMTO['Vsdsq'][metal])
            parameters['Vsd'].append(Vsd)

            # Get the metal filling
            filling = data_from_LMTO['filling'][metal]
            parameters['filling'].append(filling)

            # Store the order of the metals
            metals.append(metal)

        # Prepare the class for fitting routine 
        fitting_function =  NorskovNewnsAnderson(
            Vsd = parameters['Vsd'],
            width = parameters['width'],
            eps_a = eps_a,
            eps_sp_min = EPS_SP_MIN,
            eps_sp_max = EPS_SP_MAX,
            eps = EPS_VALUES,
            Delta0_mag = CONSTANT_DELTA0,
        )

        # Is the calculation is a restart one, choose the parameters from the last calculation
        if restart:
            previous_calc = json.load(open(f'output/{adsorbate}_parameters_{COMP_SETUP}.json'))
            alpha = previous_calc['alpha']
            beta = previous_calc['beta']
            constant_offest = previous_calc['constant_offset']
            initial_guess = [alpha, beta, constant_offest]
        else:
            initial_guess = [0.01, np.pi*0.6, 0.1]
        
        print('Initial guess: ', initial_guess)

        # Finding the fitting parameters
        data = odr.RealData(parameters['d_band_centre'], dft_energies)
        fitting_model = odr.Model(fitting_function.fit_parameters)
        fitting_odr = odr.ODR(data, fitting_model, initial_guess)
        fitting_odr.set_job(fit_type=2)
        output = fitting_odr.run()

        # Get the final hybridisation energy
        optimised_hyb = fitting_function.fit_parameters(output.beta, parameters['d_band_centre'])
        occupancies_final = np.array(fitting_function.na)[np.argsort(parameters['d_band_centre'])]
        print(f'Occupancies: {occupancies_final}', file=open(f'output/{adsorbate}_occupancies.txt', 'w'))
        print(f"d-band center: {np.sort(parameters['d_band_centre'])}", file=open(f'output/{adsorbate}_occupancies.txt', 'a'))

        # plot the parity line
        x = np.linspace(np.min(dft_energies)-0.3, np.max(dft_energies)+0.3, 2)
        ax[i].plot(x, x, '--', color='black')
        # Fix the axes to the same scale 
        ax[i].set_xlim(np.min(x), np.max(x))
        ax[i].set_ylim(np.min(x), np.max(x))

        texts = []
        for j, metal in enumerate(metals):
            # Choose the colour based on the row of the TM
            if metal in FIRST_ROW:
                colour = 'red'
            elif metal in SECOND_ROW:
                colour = 'orange'
            elif metal in THIRD_ROW:
                colour = 'green'
            ax[i].plot(dft_energies[j], optimised_hyb[j], 'o', color=colour)
            texts.append(ax[i].text(dft_energies[j], optimised_hyb[j], metal, color=colour, fontsize=14))

        adjust_text(texts, ax=ax[i]) 
        ax[i].set_aspect('equal')

        # Write out the fitted parameters as a json file
        json.dump({
            'alpha': abs(output.beta[0]),
            'beta': abs(output.beta[1]),
            'delta0': CONSTANT_DELTA0, 
            'constant_offset': output.beta[2],
            'eps_a': eps_a,
        }, open(f'output/{adsorbate}_parameters_{COMP_SETUP}.json', 'w'))

    fig.savefig(f'output/figure_3_{COMP_SETUP}.png', dpi=300)