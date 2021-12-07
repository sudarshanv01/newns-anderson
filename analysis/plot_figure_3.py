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
    # Decide the mode of running the calculation
    try:
        if sys.argv[1] == 'debug':
            debug = True
            restart = False
        elif sys.argv[1] == 'restart':
            restart = True
            debug = False
        else:
            raise ValueError('Unknown argument')
    except IndexError:
        restart = False
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
    CONSTANT_DELTA0 = 0.25
    CONSTANT_ALPHA = {'O': 0.1, 'C':0.01}
    PRECISION = 50
    print(f"Fitting parameters for adsorbate {ADSORBATES} with eps_a {EPS_A_VALUES}")

    # The functional and type of calculation we will use
    # scf only calculations in order to avoid any noise and look only for 
    # the electronic structure contribution
    FUNCTIONAL = 'PBE_scf_cold_smearing_0.2eV'
    # FUNCTIONAL = 'PBE_scf'

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{FUNCTIONAL}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{FUNCTIONAL}.json'))
    data_from_LMTO = json.load(open('inputs/modified_data_from_LMTO.json'))
    data_from_pdos_filling = json.load(open(f'output/filling_data_{FUNCTIONAL}.json'))

    # Plot the Fitted and the real adsorption energies
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    # Plot the fitted occupancies against the DFT ones
    fign, axn = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    for i in range(len(ax)):
        ax[i].set_xlabel('DFT energy (eV)')
        ax[i].set_ylabel('Hybridisation energy (eV)')
        ax[i].set_title(f'{ADSORBATES[i]}* with $\epsilon_a=$ {EPS_A_VALUES[i]} eV')
    for i in range(len(axn)):
        axn[i].set_xlabel('DFT occupancy (e)')
        axn[i].set_ylabel('NA occupancy (e)')
        axn[i].set_title(f'{ADSORBATES[i]}* with $\epsilon_a=$ {EPS_A_VALUES[i]} eV')

    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Fitting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
        # Store the parameters in order of metals in this list
        parameters = defaultdict(list)
        # Store the final DFT energies
        dft_energies = []
        metals = []
        filling_ads = []
        # If the calculation is a restart store the dict
        if restart:
            print('Restarting the fitting procedure')
            initial_guess_dict = json.load(open(f'output/{adsorbate}_parameters_{FUNCTIONAL}.json', 'r'))
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
            try:
                width = data_from_dos_calculation[metal]['width']
                parameters['width'].append(width)
                d_band_centre = data_from_dos_calculation[metal]['d_band_centre']
                parameters['d_band_centre'].append(d_band_centre)
            except KeyError:
                print('No DFT data for {}'.format(metal))
                continue

            # get the parameters from the energy calculations
            adsorption_energy = data_from_energy_calculation[adsorbate][metal]
            dft_energies.append(adsorption_energy)

            # get the idealised parameters 
            Vsd = np.sqrt(data_from_LMTO['Vsdsq'][metal])
            parameters['Vsd'].append(Vsd)

            # Get the metal filling
            filling = data_from_LMTO['filling'][metal]
            parameters['filling'].append(filling)

            # Get the pdos filling
            filling_ads.append(data_from_pdos_filling[adsorbate][metal])

            # Store the order of the metals
            metals.append(metal)

            # If restart store the initial guess
            if restart:
                parameters['initial_guess'].append(initial_guess_dict['Vak'][metal])
            else:
                parameters['initial_guess'].append(0.2*Vsd)
            
            # Store the row of the transition metal
            if metal in FIRST_ROW:
                parameters['row'].append(1)
            elif metal in SECOND_ROW:
                parameters['row'].append(2)
            elif metal in THIRD_ROW:
                parameters['row'].append(3)
            else:
                raise ValueError('Metal not in any row')

        # Fit the parameters
        fitting_function =  NorskovNewnsAnderson(
            width = parameters['width'],
            eps_a = eps_a,
            eps_sp_min = EPS_SP_MIN,
            eps_sp_max = EPS_SP_MAX,
            eps = EPS_VALUES,
            precision = PRECISION,
            Delta0_mag = CONSTANT_DELTA0, 
            alpha = CONSTANT_ALPHA[adsorbate],
            # Vsd = parameters['Vsd'],
            # row = parameters['row'],
        )

        # Finding the real coupling matrix elements
        data_to_fit = dft_energies + filling_ads
        data_base_fit = parameters['d_band_centre'] + parameters['d_band_centre']

        data = odr.RealData(x=data_base_fit, y=data_to_fit)
        fitting_model = odr.Model(fitting_function.fit_parameters)
        # Replace the initial guess such that the beta1, beta2, beta3
        # are used in the fitting procedure
        # parameters['initial_guess'] = 0.2 * np.ones(len(parameters['Vsd'])) # [ 0.2, 0.2, 0.2]
        fitting_odr = odr.ODR(data, fitting_model, parameters['initial_guess'])
        fitting_odr.set_job(fit_type=2)
        output = fitting_odr.run()
        popt = output.beta

        # Get the optimised hybridisation energy
        # and alpha from the optimised parameter list
        optimised_parameters = fitting_function.fit_parameters(popt, data_base_fit) 
        optimised_hyb = optimised_parameters[:len(dft_energies)]
        optimised_na = optimised_parameters[len(dft_energies):]

        # plot the parity line for the energies
        x = np.linspace(np.min(dft_energies)-0.3, np.max(dft_energies)+0.3, 2)
        ax[i].plot(x, x, '--', color='black')
        x_occupancy = np.linspace(0, 1, 2)
        axn[i].plot(x_occupancy, x_occupancy, '--', color='black')

        texts = []
        texts_occ = []
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
            # Plot the occupancy
            if len(optimised_na) == len(filling_ads):
                axn[i].plot(filling_ads[j], optimised_na[j], 'o', color=colour)
                texts_occ.append(axn[i].text(filling_ads[j], optimised_na[j], metal, color=colour))

        adjust_text(texts, ax=ax[i]) 
        adjust_text(texts_occ, ax=axn[i])

        # Make Vak a dictionary
        Vak = {}
        for j, metal in enumerate(metals):
            Vak[metal] = fitting_function.Vak[j]

        json.dump({
            'alpha': CONSTANT_ALPHA[adsorbate],
            'delta0': CONSTANT_DELTA0, 
            'Vak': Vak, 
            'metals': metals,
            'eps_a': eps_a,
        }, open(f'output/{adsorbate}_parameters_{FUNCTIONAL}.json', 'w'),
           indent=4)

    fig.savefig(f'output/figure_3_{FUNCTIONAL}.png', dpi=300)
    fign.savefig(f'output/figure_3_occupancy_{FUNCTIONAL}.png', dpi=300)
