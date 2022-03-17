"""Get parameters for the Newns-Anderson model and plot Figure 3 of the manuscript."""
import sys
import json
import yaml
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from scipy.optimize import minimize, least_squares, leastsq, curve_fit
from scipy import odr
from pprint import pprint
import matplotlib.pyplot as plt
from adjustText import adjust_text
from yaml import safe_load
from catchemi import NewnsAndersonLinearRepulsion, FitParametersNewnsAnderson
from create_coupling_elements import create_coupling_elements
from ase.data import covalent_radii, atomic_numbers
from ase import units
from plot_params import get_plot_params
get_plot_params()
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4

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
    EPS_VALUES = np.linspace(-30, 10, 1000)
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CONSTANT_DELTA0 = 0.1
    print(f"Fitting parameters for adsorbate {ADSORBATES} with eps_a {EPS_A_VALUES}")

    # The functional and type of calculation we will use
    # scf only calculations in order to avoid any noise and look only for 
    # the electronic structure contribution
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = 'sampled'

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{COMP_SETUP[CHOSEN_SETUP]}.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    Vsdsq_data = data_from_LMTO['Vsdsq']
    no_of_bonds = yaml.safe_load(open('inputs/number_bonds.yaml', 'r'))

    # Plot the fitted and the real adsorption energies
    fig, ax = plt.subplots(1, 2, figsize=(6.75, 3), constrained_layout=True)
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
            if isinstance(adsorption_energy, list):
                dft_energies.append(np.min(adsorption_energy))
            else:
                dft_energies.append(adsorption_energy)
            
            # Get the bond length from the LMTO calculations
            bond_length = data_from_LMTO['s'][metal]*units.Bohr #\
                        # + covalent_radii[atomic_numbers[adsorbate]] 
            bond_length_Cu = data_from_LMTO['s']['Cu']*units.Bohr # \
                        # + covalent_radii[atomic_numbers[adsorbate]]

            Vsdsq = create_coupling_elements(s_metal=s_data[metal],
                s_Cu=s_data['Cu'],
                anderson_band_width=anderson_band_width_data[metal],
                anderson_band_width_Cu=anderson_band_width_data['Cu'],
                r=bond_length,
                r_Cu=bond_length_Cu,
                normalise_bond_length=True,
                normalise_by_Cu=True)
            # Report the square root
            Vsd = np.sqrt(Vsdsq)
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

        # Is the calculation is a restart one, choose the parameters from the last calculation
        if restart:
            previous_calc = json.load(open(f'output/{adsorbate}_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json'))
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

        # plot the parity line
        x = np.linspace(np.min(dft_energies)-0.6, np.max(dft_energies)+0.6, 2)
        ax[i].plot(x, x, '--', color='tab:grey', linewidth=1)
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
            texts.append(ax[i].text(dft_energies[j], optimised_hyb[j], metal, color=colour, ))

        adjust_text(texts, ax=ax[i]) 
        ax[i].set_aspect('equal')

        # Write out the fitted parameters as a json file
        json.dump({
            'alpha': abs(output.beta[0]),
            'beta': abs(output.beta[1]),
            'delta0': CONSTANT_DELTA0, 
            'constant_offset': output.beta[2],
            'eps_a': eps_a,
            # 'no_of_bonds': no_of_bonds,
        }, open(f'output/{adsorbate}_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json', 'w'))

    fig.savefig(f'output/figure_3_fitting_{COMP_SETUP[CHOSEN_SETUP]}.png', dpi=300)