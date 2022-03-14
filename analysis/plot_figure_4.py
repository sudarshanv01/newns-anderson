"""Plot an idealised scaling model for C and O scaling."""
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

def create_plot_layout():
    """Create a plot layout for plotting the Newns-Anderson
    dos and the energies of orthogonalisation, spd hybridisation
    energy for each specific adsorbate."""

    fig, ax = plt.subplots(2, 3, figsize=(5, 3), constrained_layout=True)
    for i, a in enumerate(ax):
        if i == 1:
            a[0].set_xlabel(r'$\epsilon_d$ (eV)')
            a[1].set_xlabel(r'$\epsilon_d$ (eV)')
            a[2].set_xlabel(r'$\epsilon_a$ (eV)')
        a[0].set_ylabel(r'$E_{\rm hyb}$ %s* (eV)'%ADSORBATES[i])
        a[1].set_ylabel(r'$E_{\rm hyb} ^\prime$ %s*'%ADSORBATES[i])
        # Make a twin axis for the hybridisation energy panel
    ax[1,2].set_ylabel(r'$\epsilon_s$ (eV)')
    ax[0,2].axis('off')


    # Confirm the density of states figures
    # from the Newns-Anderson equations
    figs = plt.figure(figsize=(10,11), constrained_layout=True)
    gs = figs.add_gridspec(nrows=6, ncols=6)
    axi1 = figs.add_subplot(gs[:, 0:2])
    axi2 = figs.add_subplot(gs[:, 2:4])
    axi3 = figs.add_subplot(gs[:, 4:6])
    # Set the axes labels
    axi1.set_xlabel(r'$\epsilon - \epsilon_{f}$ (eV)')
    axi2.set_xlabel(r'$\epsilon - \epsilon_{f}$ (eV)')
    axi3.set_xlabel(r'$\epsilon - \epsilon_{f}$ (eV)')
    axi1.set_ylabel('Projected Density of States')
    axi1.set_title('3d')
    axi2.set_title('4d')
    axi3.set_title('5d')
    # Remove y-ticks from 4, 5, 6
    axi1.set_yticks([])
    axi2.set_yticks([])
    axi3.set_yticks([])

    return fig, figs, ax, np.array([axi1, axi2, axi3])

def set_same_limits(axes, y_set=True, x_set=False):
    """Set the limits of all axes to the same value."""
    if y_set:
        ylims = []
        for i in range(len(axes)):
            ylims.append(axes[i].get_ylim())
        ylims = np.array(ylims).T
        # get the minimum and maximum 
        ymin = np.min(ylims[0])
        ymax = np.max(ylims[1])
        for ax in axes:
            ax.set_ylim([ymin, ymax])
    if x_set:
        xlims = []
        for i in range(len(axes)):
            xlims.append(axes[i].get_xlim())
        xlims = np.array(xlims).T
        # get the minimum and maximum
        xmin = np.min(xlims[0])
        xmax = np.max(xlims[1])
        for ax in axes:
            ax.set_xlim([xmin, xmax])
    
def create_range_parameters(param1, param2, n=50):
    """Linearly interpolate between two parameters."""
    return np.linspace(param1, param2, n)

def normalise_na_quantities(quantity, x_add):
    """Utility function to align the density of states for Newns-Anderson plots."""
    return quantity / np.max(np.abs(quantity)) + x_add

def plot_epsa_value(eps_a):
    """Function to evaluate if eps_a value is to 
    be plotted."""
    chosen_values = [o_parameters['eps_a'], c_parameters['eps_a']]
    if eps_a in chosen_values: 
        return True, chosen_values.index(eps_a) 
    else:
        return False, None

if __name__ == '__main__':
    """Plot the hybridisation energy as a function of the d-band
    centre for 3d, 4d and 5d metals. Change eps_a to see when the curve
    levels off. Assume that the fractional filling depends linearly on
    the d-band centre. This assumption is used so that we can get smooth 
    variations in the figure."""

    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = 'energy'
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        c_parameters = json.load(f)
    with open(f"output/fitting_metal_parameters.json", 'r') as f:
        metal_parameters = json.load(f)
    GRID_LEVEL = 'high' # 'high' or 'low'

    # Create range of parameters 
    if GRID_LEVEL == 'high':
        NUMBER_OF_ADSORBATES = 25
        NUMBER_OF_METALS = 50
        GRID_SPACING_DERIV = 120
    elif GRID_LEVEL == 'low':
        NUMBER_OF_ADSORBATES = 2
        NUMBER_OF_METALS = 4
        GRID_SPACING_DERIV = 10
    EPS_RANGE = np.linspace(-20, 20, 1000)
    ADSORBATES = ['O', 'C']

    # Fix the energy width of the sp 
    # states of the metal
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CUTOFF_VALUE = -0.1

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

    # Collect the positions of the locations where the 
    # newns-anderson energy levels-off measured through
    # the derivative of the hybrisation energy with eps_d 
    eps_s = defaultdict(list)

    # Input parameters to help with the dos from Newns-Anderson
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{COMP_SETUP[CHOSEN_SETUP]}.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    dft_Vsdsq = json.load(open(f"output/dft_Vsdsq.json"))

    # Get the main figure with the energies and the axes
    # and the supporting figure with the density of states
    fig, figs, ax, axs = create_plot_layout() 

    # get a color cycle for the different adsorbates based on viridis
    color = plt.cm.coolwarm_r(np.linspace(0, 1, NUMBER_OF_ADSORBATES))
    color_row = ['tab:red', 'tab:blue', 'tab:green',]
    marker_row = ['o', 's', '^']
    ls_row = ['-', '-', '-']
    # Plot the legend for the rows
    for i in range(len(color_row)):
        ax[0,2].plot([], [], color=color_row[i], ls=ls_row[i], label=f'{i+3}' + r'$d$')
    ax[0,2].legend(loc='upper right')

    # Store the final energies to plot in a scaling line
    final_energy_scaling = defaultdict(lambda: defaultdict(list))

    for a, eps_a in enumerate(eps_a_range):
        # Iterate separately over the different adsorbates
        # Adsorbate specific parameters are computed here.
        alpha = alpha_range[a]
        beta  = beta_range[a]
        delta0 = delta0_range[a]
        constant = constant_range[a]
        # Decide is the eps_a value is to be plotted
        to_plot, index_plot = plot_epsa_value(eps_a)

        for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            # Create a plot of hybridisation energy as a function of
            # the d-band centre where each curve is for a different 
            # row of the metal.

            # get the metal fitting parameters
            Vsd_fit = metal_parameters['Vsd'][str(j)]
            wd_fit = metal_parameters['width'][str(j)]
            epsd_filling_fit = metal_parameters['epsd_filling'][str(j)]

            # Consider only a specific range of metals in the analysis
            # Those used in Figures 1-3 of the paper
            filling_min, filling_max = metal_parameters['filling_minmax'][str(j)]
            eps_d_min, eps_d_max = metal_parameters['eps_d_minmax'][str(j)]
            # Generate grid for differentiation
            diff_grid = np.linspace(eps_d_min, eps_d_max, GRID_SPACING_DERIV)

            # Create linearlised variation of this parameter for
            # the chosen set of materials
            filling_range = np.linspace(filling_max, filling_min, NUMBER_OF_METALS)
            eps_d_range = np.linspace(eps_d_min, eps_d_max, NUMBER_OF_METALS)


            # Functions of Vsd and width as a function of the filling
            function_Vsd, function_Vsd_p = get_fitted_function('Vsd') 
            function_wd, function_wd_p = get_fitted_function('wd')
            # Generate the kwargs needed to pass in eps_d values 
            # to functions which are fit to the filling  
            kwargs = {'input_epsd':True, 'fitted_epsd_to_filling':epsd_filling_fit}
            kwargs_deriv = {'input_epsd':True, 
                            'fitted_epsd_to_filling':epsd_filling_fit, 
                            'is_derivative':True}

            # Store the metal parameters to be used in the fitting model
            parameters_metal = defaultdict(list)

            for i, filling in enumerate(filling_range):
                # Continuous setting of parameters for each 
                # continous variation of the metal
                Vsd = function_Vsd(filling, *Vsd_fit )
                width = function_wd(filling, *wd_fit )
                eps_d = eps_d_range[i]                

                parameters_metal['Vsd'].append(Vsd)
                parameters_metal['eps_d'].append(eps_d)
                parameters_metal['width'].append(width)
                parameters_metal['filling'].append(filling)

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
            )
            jna = FitParametersNewnsAnderson(**kwargs_fitting)

            # Gather energies
            chemisorption_energy = jna.fit_parameters( [alpha, beta, constant], eps_d_range)
            # This energy will be used as a basis for comparison
            # We are looking for the maximum eps_d at which energy_to_plot
            # saturates to its values of -Delta0
            energy_to_plot = jna.hyb_energy
            energy_to_plot = np.array(energy_to_plot)
            ortho_energy = jna.ortho_energy
            occupancy = jna.occupancy

            if to_plot: 
                # If the adsorbate is C or O, plot the energy
                # ax[index_plot, 0].plot(parameters_metal['eps_d'], chemisorption_energy,
                #             '-', color=color_row[j], ls=ls_row[j],)
                ax[index_plot, 0].plot(parameters_metal['eps_d'], 
                                 energy_to_plot, '-',
                                 color=color_row[j], ls=ls_row[j],)
        
            # Compute the derivative of the hybridisation energy with 
            # the d-band centre to be plotted in the third figure
            # Generate the continuous variation of parameters as a 
            # function of eps_d for each row.
            f_Vsd = lambda x: function_Vsd(x, *Vsd_fit, **kwargs)
            f_Vsd_p = lambda x: function_Vsd_p(x, *Vsd_fit, **kwargs_deriv)
            f_wd = lambda x: function_wd(x, *wd_fit, **kwargs)
            f_wd_p = lambda x: function_wd_p(x, *wd_fit, **kwargs_deriv)

            # Get the derivative of the hybridisation energy with eps_d
            derivative = NewnsAndersonDerivativeEpsd(f_Vsd=f_Vsd,f_Vsd_p=f_Vsd_p,
                                                    eps_a=eps_a, eps=EPS_RANGE,
                                                    f_wd=f_wd, f_wd_p=f_wd_p,
                                                    diff_grid=diff_grid,
                                                    alpha=alpha, beta=beta,
                                                    Delta0_mag=delta0,
                                                    constant_offset=constant)
            analytical_hyb_deriv = derivative.get_hybridisation_energy_prime_epsd()

            # Store the highest epsilon_d value at which the 
            # analytical derivative saturates to CUTOFF_VALUE
            saturation_epsd_arg = [ a for a in range(len(analytical_hyb_deriv)) 
                                if analytical_hyb_deriv[a] > CUTOFF_VALUE]
            if len(saturation_epsd_arg) > 0:
                saturation_epsd = diff_grid[saturation_epsd_arg]            
                eps_s[j].append(np.max(saturation_epsd))
                if to_plot:
                    # ax[index_plot,0].axvline(x=np.max(saturation_epsd),
                    #                          color=color_row[j],
                    #                          alpha=0.5,
                    #                          ls='--')
                    ax[index_plot,1].axvline(x=np.max(saturation_epsd),
                                             color=color_row[j],
                                             alpha=0.5,
                                             ls='--')
            else:
                eps_s[j].append(None)

            if to_plot:
                # Plot the derivative of the hybridisation energy with eps_d
                ax[index_plot, 1].plot(diff_grid, analytical_hyb_deriv,
                                       ls=ls_row[j], color=color_row[j])
    # Plot the eps_s values for each row
    for i, (row, eps_s_row) in enumerate(eps_s.items()):
        if eps_s_row:
            ax[1,2].plot(eps_a_range, eps_s_row, ls=ls_row[j], color=color_row[i]) 
             

    # Add figure numbers
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(ax.T.flatten()):
        if i == 4:
            continue
        if i == 5:
            i -= 1
        a.annotate(alphabet[i]+')', xy=(0.05, 0.5), fontsize=8, xycoords='axes fraction')

    fig.savefig(f'output/figure_4_{COMP_SETUP}.png', dpi=300)
    figs.savefig(f'output/final_param_na_dos_{COMP_SETUP}.png', dpi=300)
    