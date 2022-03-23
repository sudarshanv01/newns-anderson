"""Plot the orthogonolisation energies for different values of eps_a."""
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
import pickle
from catchemi import ( NewnsAndersonLinearRepulsion,
                       NewnsAndersonNumerical,
                       NewnsAndersonDerivativeEpsd,
                       FitParametersNewnsAnderson
                     )  
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl']
EPSILON = 1e-2

def create_range_parameters(param1, param2, n=50):
    """Linearly interpolate between two parameters."""
    return np.linspace(param1, param2, n)

def normalise_na_quantities(quantity, x_add):
    """Utility function to align the density of states for Newns-Anderson plots."""
    return quantity / np.max(np.abs(quantity)) + x_add

if __name__ == '__main__':
    """Compute the orthogonalisation energies for the different 
    values of eps_a varying between C* and O* and see if they 
    scale with one another."""

    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = open('chosen_setup', 'r').read() 
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        c_parameters = json.load(f)
    GRID_LEVEL = 'low' # 'high' or 'low'

    # Create range of parameters 
    if GRID_LEVEL == 'high':
        NUMBER_OF_ADSORBATES = 20
        NUMBER_OF_METALS = 30
    elif GRID_LEVEL == 'low':
        NUMBER_OF_ADSORBATES = 4 
        NUMBER_OF_METALS = 8 
    EPS_RANGE = np.linspace(-30, 10, 1000)
    ADSORBATES = ['O', 'C']

    # Fix the energy width of the sp 
    # states of the metal
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15

    # Create an idealised range of parameters by interpolating each 
    # parameter inclusing eps_a linearly over the entire energy range
    # This script assumes that we are going from O (at -5 eV) to 
    # C (at -1 eV), lower to higher energies.
    eps_a_range = create_range_parameters(o_parameters['eps_a'], 
                                          c_parameters['eps_a'],
                                          n=NUMBER_OF_ADSORBATES)
    alpha_range = create_range_parameters(o_parameters['alpha'],
                                          c_parameters['alpha'],
                                          n=NUMBER_OF_ADSORBATES)
    beta_range  = create_range_parameters(o_parameters['beta'], 
                                          c_parameters['beta'],
                                          n=NUMBER_OF_ADSORBATES)
    delta0_range = create_range_parameters(o_parameters['delta0'],
                                           c_parameters['delta0'],
                                           n=NUMBER_OF_ADSORBATES)
    constant_range = create_range_parameters(o_parameters['constant_offset'], 
                                             c_parameters['constant_offset'], 
                                             n=NUMBER_OF_ADSORBATES)

    # Input parameters to help with the dos from Newns-Anderson
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{COMP_SETUP[CHOSEN_SETUP]}.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    dft_Vsdsq = json.load(open(f"output/dft_Vsdsq.json"))
    no_of_bonds = yaml.safe_load(open('inputs/number_bonds.yaml', 'r'))
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    minmax_parameters = json.load(open('output/minmax_parameters.json'))

    with open(f"output/spline_objects.pkl", 'rb') as f:
        spline_objects = pickle.load(f)

    # Get the main figure with the energies and the axes
    # and the supporting figure with the density of states
    fig, ax = plt.subplots(1, NUMBER_OF_ADSORBATES,
                           figsize=(2.25*NUMBER_OF_ADSORBATES, 2.5),
                           sharex=True, sharey=True,
                           constrained_layout=True)

    # get a color cycle for the different adsorbates based on viridis
    color = plt.cm.coolwarm_r(np.linspace(0, 1, NUMBER_OF_ADSORBATES))
    color_row = ['tab:red', 'tab:blue', 'tab:green',]
    marker_row = ['o', 's', '^']
    ls_row = ['-', '-', '-']

    # Store the final energies to plot in a scaling line
    final_energy_scaling = defaultdict(lambda: defaultdict(list))

    for a, eps_a in enumerate(eps_a_range):
        # Iterate separately over the different adsorbates
        # Adsorbate specific parameters are computed here.
        alpha = alpha_range[a]
        beta  = beta_range[a]
        delta0 = delta0_range[a]
        constant = constant_range[a]

        for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            # Create a plot of hybridisation energy as a function of
            # the d-band centre where each curve is for a different 
            # row of the metal.

            # get the metal fitting parameters
            s_fit = spline_objects[j]['s']
            Delta_anderson_fit = spline_objects[j]['Delta_anderson']
            wd_fit = spline_objects[j]['width']
            eps_d_fit = spline_objects[j]['eps_d']


            # Consider only a specific range of metals in the analysis
            # Those used in Figures 1-3 of the paper
            filling_min, filling_max = minmax_parameters[str(j)]['filling']
            eps_d_min, eps_d_max = minmax_parameters[str(j)]['eps_d']

            # Create linearlised variation of this parameter for
            # the chosen set of materials
            filling_range = np.linspace(filling_max, filling_min, NUMBER_OF_METALS)
            eps_d_range = np.linspace(eps_d_min, eps_d_max, NUMBER_OF_METALS)

            # Store the metal parameters to be used in the fitting model
            parameters_metal = defaultdict(list)

            for i, filling in enumerate(filling_range):
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
                # TODO: Fix the number of bonds issue
                parameters_metal['no_of_bonds'].append(no_of_bonds[CHOSEN_SETUP]['average'])

            # Now plot the energies for each row
            kwargs_fitting = dict(
                Vsd = parameters_metal['Vsd'],
                eps_a = eps_a,
                width = parameters_metal['width'],
                eps = EPS_RANGE, 
                eps_sp_max=EPS_SP_MAX,
                eps_sp_min=EPS_SP_MIN,
                Delta0_mag=delta0,
                store_hyb_energies = True,
                no_of_bonds = parameters_metal['no_of_bonds'],
            )
            jna = FitParametersNewnsAnderson(**kwargs_fitting)

            # Gather energies
            chemisorption_energy = jna.fit_parameters( [alpha, beta, constant], eps_d_range)
            hybridisation_energy = jna.hyb_energy
            ortho_energy = jna.ortho_energy
            occupancy = jna.occupancy
            filling_factor = jna.filling_factor

            final_energy_scaling[eps_a][j].extend(ortho_energy)
    
    # Plot the energies for for each eps_a with the eps_a for O
    for a, eps_a in enumerate(eps_a_range):
        for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            # Plot the energies for each row
            ax[a].plot(final_energy_scaling[-5][j], final_energy_scaling[eps_a][j],
                        color=color_row[j],
                        marker='o',
                        ls=ls_row[j],
                        label=f"{metal_row}",
                        markersize=5,
                        )
        ax[a].set_xlabel("$\Delta E_{\mathrm{ortho}}$ for $\epsilon_a=-5.0$ eV (eV)")
        ax[a].set_ylabel("$\Delta E_{\mathrm{ortho}}$ for $\epsilon_a={%1.1f}$ eV (eV)"%eps_a)

    fig.savefig(f'output/compare_ortho_energies_{COMP_SETUP[CHOSEN_SETUP]}.png', dpi=300)
    