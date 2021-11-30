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
    # Check if we should run in debug mode
    try:
        if sys.argv[1] == 'debug':
            debug = True
        else:
            debug = False
    except IndexError:
        debug = False

    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    if debug:
        KEEP_LIST = ['V', 'Cr', 'Cu', 'Au', 'Ag', 'Ir', 'Pt', 'Os']
    else:
        KEEP_LIST = []

    # Choose a sequence of adsorbates
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1 ] # eV
    EPS_VALUES = np.linspace(-20, 20, 1000)
    EPS_SP_MIN = -20
    EPS_SP_MAX = 20
    # CONSTANT_DELTA0 = 3.0
    # CONSTANT_ALPHA = 0.075
    print(f"Fitting parameters for adsorbate {ADSORBATES} with eps_a {EPS_A_VALUES}")

    # The functional and type of calculation we will use
    # scf only calculations in order to avoid any noise and look only for 
    # the electronic structure contribution
    FUNCTIONAL = 'PBE_scf'

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{FUNCTIONAL}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{FUNCTIONAL}.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    data_fitting_pdos = json.load(open(f'output/filling_data_{FUNCTIONAL}.json'))

    # Plot the Fitted and the real adsorption energies
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    fign, axn = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    for i in range(len(ax)):
        ax[i].set_xlabel('DFT energy (eV)')
        ax[i].set_ylabel('Hybridisation energy (eV)')
        ax[i].set_title(f'{ADSORBATES[i]}* with $\epsilon_a=$ {EPS_A_VALUES[i]} eV')
    for i in range(len(axn)):
        axn[i].set_xlabel('DFT occupancy (eV)')
        axn[i].set_ylabel('Newns-Anderson occupancy (eV)')

    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Fitting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
        # Store the parameters in order of metals in this list
        parameters = defaultdict(list)
        # Store the final DFT energies
        dft_energies = []
        metals = []
        filling_ads = []
        # Store quantities for different sources that could be 
        # useful for fitting the DFT energies
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

            # Get the adsorbate filling data
            parameters['filling_adsorbate'].append(data_fitting_pdos[adsorbate][metal])
            filling_ads.append(data_fitting_pdos[adsorbate][metal])

            # Store the order of the metals
            metals.append(metal)

        # Fit the parameters
        fitting_function =  NorskovNewnsAnderson(
            width = parameters['width'],
            eps_a = eps_a,
            eps_sp_min = EPS_SP_MIN,
            eps_sp_max = EPS_SP_MAX,
            eps = EPS_VALUES,
        )

        # Initial guesses to the fitting procedure 
        # is that using the LMTO Vsd parameters.
        initial_guess = 0.2 * np.array(parameters['Vsd'])
        initial_guess_alpha = [0.1]
        initial_guess_Delta0 = [1.0]
        initial_guess = np.concatenate((initial_guess,
                                        initial_guess_Delta0,
                                        initial_guess_alpha,
                                        ))

        # Finding the real coupling matrix elements
        data_to_fit = dft_energies + filling_ads
        data_base_fit = parameters['d_band_centre'] + parameters['d_band_centre']

        data = odr.RealData(x=data_base_fit, y=data_to_fit)
        fitting_model = odr.Model(fitting_function.fit_parameters)
        fitting_odr = odr.ODR(data, fitting_model, initial_guess)
        fitting_odr.set_job(fit_type=2)
        output = fitting_odr.run()
        popt = output.beta

        # Store common alpha, this parameter is the 
        # same for all the adsorbates
        # alpha = fitting_function.alpha

        # Get the optimised hybridisation energy
        # and alpha from the optimised parameter list
        optimised_parameters = fitting_function.fit_parameters(popt, data_base_fit) 
        optimised_hyb = optimised_parameters[:len(dft_energies)]
        optimised_na = optimised_parameters[len(dft_energies):] 

        # plot the parity line for the energies
        x = np.linspace(np.min(dft_energies)-0.3, np.max(dft_energies)+0.3, 2)
        ax[i].plot(x, x, '--', color='black')
        xn = np.linspace(0, 1, 2)
        axn[i].plot(xn, xn, '--', color='black')

        texts = []
        texts_n = []
        for j, metal in enumerate(metals):
            # Choose the colour based on the row of the TM
            if metal in FIRST_ROW:
                colour = 'red'
            elif metal in SECOND_ROW:
                colour = 'orange'
            elif metal in THIRD_ROW:
                colour = 'green'
            ax[i].plot(dft_energies[j], optimised_hyb[j], 'o', color=colour)
            texts.append(ax[i].text(dft_energies[j], optimised_hyb[j], metal, color=colour))
            # Plot the converged occupancies
            axn[i].plot(filling_ads[j], optimised_na[j], 'o', color=colour)
            texts_n.append(axn[i].text(filling_ads[j], optimised_na[j], metal, color=colour))

        adjust_text(texts, ax=ax[i]) 
        adjust_text(texts_n, ax=axn[i])

        # Write out the fitted parameters as a json file
        # Create a dictionary of metal and fitted Vak parameters
        fit_Vak = {}
        for i, metal in enumerate(metals):
            fit_Vak[metal] = abs(popt[i])
        json.dump({
            'alpha': abs(popt[-1]), 
            'delta0': abs(popt[-2]), 
            'Vak': fit_Vak, 
            'metals': metals,
            'eps_a': eps_a,
        }, open(f'output/{adsorbate}_parameters_{FUNCTIONAL}.json', 'w'),
           indent=4)

    fig.savefig(f'output/figure_3_{FUNCTIONAL}.png', dpi=300)
    fign.savefig(f'output/figure_3_unbound__{FUNCTIONAL}.png', dpi=300)
