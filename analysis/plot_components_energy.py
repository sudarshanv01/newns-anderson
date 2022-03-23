"""Plot the different components of the energy for C and O."""
import numpy as np
import json
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
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
import pickle
from scipy import stats
from plot_figure_4 import set_same_limits
from plot_params import get_plot_params
get_plot_params()
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4

C_COLOR = 'tab:purple'
O_COLOR = 'tab:orange'

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 
ls_ads = ['--', '-']

if __name__ == '__main__':
    """Plot the model chemisorption energies, hybridisation
    energies and the orthogonalisation energies for C and O."""

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
    fig, ax = plt.subplots(len(ADSORBATES)+1, 3, figsize=(6.75, 6.75)) 
    # Create a twin axis for the occupancy
    ax2 = []
    for i, a in enumerate(ax[:-1,-1]):
        ax2.append(a.twinx())
    ax2 = np.array(ax2)

    # Store the scaling energies for each adsorbate
    scaling_energies = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

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
            store_hyb_energies=True,
        )

        fitting_function =  FitParametersNewnsAnderson(**kwargs_fit)

        # Get the final hybridisation energy
        optimised_hyb = fitting_function.fit_parameters(final_params, parameters['d_band_centre'])

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
            hybridisation_energy = fitting_function.hyb_energy
            occupancy = fitting_function.occupancy
            filling_factor = fitting_function.filling_factor
            na_plus_f = occupancy + filling_factor
            ortho_energy = fitting_function.ortho_energy
            ax[i][0].plot(eps_d_range, chemisorption_energy, color=color_row[j]) 
            ax[i][1].plot(eps_d_range, hybridisation_energy, color=color_row[j]) 
            ax[i][2].plot(eps_d_range, ortho_energy, color=color_row[j]) 
            ax2[i].plot(eps_d_range, na_plus_f, 'o', color=color_row[j], alpha=0.5)

            # Store the chemisorption energies in scaling_energies
            # to plot a scaling line
            scaling_energies['chem'][j][adsorbate] = chemisorption_energy
            scaling_energies['hyb'][j][adsorbate] = hybridisation_energy
            scaling_energies['ortho'][j][adsorbate] = ortho_energy

        ax[i,0].set_ylabel(r'$\Delta E_{\mathrm{chem}}$ %s / eV'%adsorbate)
        ax[i,1].set_ylabel(r'$\Delta E_{\mathrm{hyb}}$ %s / eV'%adsorbate)
        ax[i,2].set_ylabel(r'$\Delta E_{\mathrm{ortho}}$ %s (-) / eV'%adsorbate)
        ax2[i].set_ylabel(r'$\left ( n_a + f \right )$ %s (o) / e'%adsorbate)

    # Plot the scaling line
    for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        ax[i+1][0].plot(scaling_energies['chem'][j]['C'], scaling_energies['chem'][j]['O'], color=color_row[j])
        ax[i+1][1].plot(scaling_energies['hyb'][j]['C'], scaling_energies['hyb'][j]['O'], color=color_row[j])
        ax[i+1][2].plot(scaling_energies['ortho'][j]['C'], scaling_energies['ortho'][j]['O'], color=color_row[j])

    ax[-1,0].set_xlabel(r'$\Delta E_{\mathrm{chem}}$ C / eV')
    ax[-1,0].set_ylabel(r'$\Delta E_{\mathrm{chem}}$ O / eV')
    ax[-1,1].set_xlabel(r'$\Delta E_{\mathrm{hyb}}$ C / eV')
    ax[-1,1].set_ylabel(r'$\Delta E_{\mathrm{hyb}}$ O / eV')
    ax[-1,2].set_xlabel(r'$\Delta E_{\mathrm{ortho}}$ C / eV')
    ax[-1,2].set_ylabel(r'$\Delta E_{\mathrm{ortho}}$ O / eV')
    
    
    for a in ax[:-1,0]:
        a.set_xlabel('$\epsilon_d$ / eV ')
    for a in ax[:-1,1]:
        a.set_xlabel('$\epsilon_d$ / eV')
    for a in ax[:-1,2]:
        a.set_xlabel('$\epsilon_d$ / eV')
    
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(ax.flatten()):
        a.annotate(alphabet[i]+')', xy=(0.05, 0.6), xycoords='axes fraction')

    fig.tight_layout()
    fig.savefig(f'output/components_energy_{COMP_SETUP[CHOSEN_SETUP]}.png', dpi=300)

