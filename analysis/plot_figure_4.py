"""Plot an idealised scaling model for C and O scaling."""
import numpy as np
import scipy
import json
import yaml
from norskov_newns_anderson.NewnsAnderson import NorskovNewnsAnderson, NewnsAndersonNumerical
from collections import defaultdict
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from scipy.optimize import curve_fit
from scipy import special
import string
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl']
EPSILON = 1e-2

def create_plot_layout():
    """Create a plot layout for plotting the Newns-Anderson
    dos and the energies of orthogonalisation, spd hybridisation
    energy for each specific adsorbate."""
    fig = plt.figure(figsize=(8,6), constrained_layout=True)
    gs = fig.add_gridspec(nrows=6, ncols=6)
    # The first 2 rows will be the orthogonalisation energies, spd
    # hybridisation energy and the total energy as a function of the
    # d-band centre.
    ax10 = fig.add_subplot(gs[0:3, 0:3])
    ax11 = fig.add_subplot(gs[0:3, 3:6])

    ax2 = fig.add_subplot(gs[3:6, 0:3])
    ax3 = fig.add_subplot(gs[3:6, 3:6])
    
    # Set the axes labels
    ax10.set_xlabel('$\epsilon_{d} - \epsilon_{F}$ (eV)')
    ax11.set_xlabel('$\epsilon_{d} - \epsilon_{F}$ (eV)')
    ax2.set_xlabel('$\epsilon_{d} - \epsilon_{F}$ (eV)')
    # Each of ax1, ax2 and ax3 have a twinx() axis, which is used to
    # plot the derivative
    ax11.set_ylabel('$\Delta E_{\mathregular{hyb}}$ / eV (--)')

    figs = plt.figure(figsize=(10,11), constrained_layout=True)
    gs = figs.add_gridspec(nrows=6, ncols=6)

    # Then make three plots with the density of states coming from
    # the different solutions of the Newns-Anderson equation.
    ax4 = figs.add_subplot(gs[:, 0:2])
    ax5 = figs.add_subplot(gs[:, 2:4])
    ax6 = figs.add_subplot(gs[:, 4:6])
    # Set the axes labels
    ax4.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax5.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax6.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax4.set_ylabel('Projected Density of States')
    ax4.set_title('3d')
    ax5.set_title('4d')
    ax6.set_title('5d')
    # Remove y-ticks from 4, 5, 6
    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])

    return fig, figs, np.array([ [np.array([ax10, ax11]), ax2, ax3], [ax4, ax5, ax6] ], dtype=object)

def create_aux_plot_layout():
    """Create the plot layout for the descriptor as a 
    function of the eps_a."""
    fig, ax = plt.subplots(1, 1, figsize=(6,5), constrained_layout=True)
    ax.set_xlabel(r'$\epsilon_{a}$ (eV)')
    ax.set_ylabel(r'$a$ (eV$^{-1}$)')
    return fig, ax

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

class FittingEnergyVariation:
    def __init__(self, filling, Vsdsq_fit, beta):
        self.filling = filling
        self.Vsdsq_fit = Vsdsq_fit
        self.beta = beta

    def functional_form_energy(self, eps_d, a, b, c):
        """Fitting function for the functional form of the energy."""
        eps_d = np.array(eps_d)
        Vsdsq = func_a_by_r(self.filling, *self.Vsdsq_fit)
        return a * self.beta * Vsdsq / (-1*eps_d) + b * eps_d  + c 

def func_a_by_r(x, a):
    return a / x

def func_a_r_sq(x, a, b, c):
    return b - a * ( x - c) **2

if __name__ == '__main__':
    """Plot the hybridisation energy as a function of the d-band
    centre for 3d, 4d and 5d metals. Change eps_a to see when the curve
    levels off. Assume that the fractional filling depends linearly on
    the d-band centre. This assumption is used so that we can get smooth 
    variations in the figure."""

    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))['group'][0]
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{COMP_SETUP}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{COMP_SETUP}.json", 'r') as f:
        c_parameters = json.load(f)
    with open(f"output/fitting_metal_parameters_{COMP_SETUP}.json", 'r') as f:
        metal_parameters = json.load(f)

    # Create range of parameters 
    NUMBER_OF_ADSORBATES = 20
    NUMBER_OF_METALS = 200
    PLOT_METAL_DOS = 2
    EPS_RANGE = np.linspace(-15, 15, 1000)
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15

    # Create an idealised range of parameters
    eps_a_range = create_range_parameters(o_parameters['eps_a'], c_parameters['eps_a'], n=NUMBER_OF_ADSORBATES)
    alpha_range = create_range_parameters(o_parameters['alpha'], c_parameters['alpha'], n=NUMBER_OF_ADSORBATES)
    beta_range  = create_range_parameters(o_parameters['beta'], c_parameters['beta'], n=NUMBER_OF_ADSORBATES)
    delta0_range = create_range_parameters(o_parameters['delta0'], c_parameters['delta0'], n=NUMBER_OF_ADSORBATES)
    constant_range = create_range_parameters(o_parameters['constant_offset'], 
                                             c_parameters['constant_offset'], 
                                             n=NUMBER_OF_ADSORBATES)

    # Input parameters to help with the dos from Newns-Anderson
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{COMP_SETUP}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{COMP_SETUP}.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Each column is for a different row of transition metals
    fig, figs, ax = create_plot_layout() 

    # get a color cycle for the different adsorbates based on viridis
    color = plt.cm.coolwarm_r(np.linspace(0, 1, NUMBER_OF_ADSORBATES))
    color_row = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    marker_row = ['o', 's', '^']

    # Store the final energies to plot in a scaling line
    final_energy_scaling = defaultdict(lambda: defaultdict(list))

    for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        # Create a plot of hybridisation energy as a function of the d-band centre.
        # where each plot is for a different row of the metal.
        # get the metal fitting parameters
        Vsdsq_fit = metal_parameters['Vsdsq'][str(j)]
        wd_fit = metal_parameters['width'][str(j)]
        filling_min, filling_max = metal_parameters['filling_minmax'][str(j)]
        eps_d_min, eps_d_max = metal_parameters['eps_d_minmax'][str(j)]
        filling_range = np.linspace(filling_max, filling_min, NUMBER_OF_METALS)
        eps_d_range = np.linspace(eps_d_min, eps_d_max, NUMBER_OF_METALS)

        # Collect the total energies for each adsorbate
        total_energy_adsorbate = []

        # Collect the positions of the maximum of the derivative
        argmax_derivative = []

        for a, eps_a in enumerate(eps_a_range):
            # Each line for different rows of metals is for a different adsorbate
            # get the parameters for each adsorbate
            alpha = alpha_range[a]
            beta  = beta_range[a]
            delta0 = delta0_range[a]
            constant = constant_range[a]

            # Store the metal parameters to be used in the JNA model
            parameters_metal = defaultdict(list)

            for i, filling in enumerate(filling_range):
                # Continuous setting of parameters
                Vsd = np.sqrt(func_a_by_r( filling, *Vsdsq_fit ) )
                eps_d = eps_d_range[i]                
                width = func_a_r_sq(filling, *wd_fit)

                # Iterate over each metal to get the metal specific parameters
                parameters_metal['Vsd'].append(Vsd)
                parameters_metal['eps_d'].append(eps_d)
                parameters_metal['width'].append(width)
                parameters_metal['filling'].append(filling)

                # Plot the projected density of states from the Newns-Anderson model
                # for these parameters for this adsorbates
                hybridisation = NewnsAndersonNumerical(
                    Vak = Vsd,
                    eps_a = eps_a,
                    eps_d = eps_d,
                    width = width,
                    eps = EPS_RANGE,     
                    Delta0_mag = delta0,
                    eps_sp_max=EPS_SP_MAX,
                    eps_sp_min=EPS_SP_MIN,
                )
                hybridisation.calculate_energy()
            
                # Decide on the x-position based on the d-band centre
                denote_pos = NUMBER_OF_METALS / PLOT_METAL_DOS

                # Store the occupied energy index, where eps < 0
                occupied_energy_index = np.where(hybridisation.eps < 0)[0]

                # Get the metal projected density of states
                # Pick every PLOT_METAL_DOS number of points
                if i % PLOT_METAL_DOS == 0:
                    x_pos =  2.5 * i / denote_pos
                    # Get the Delta from the calculation
                    Delta = hybridisation.get_Delta_on_grid()
                    Delta = normalise_na_quantities( Delta, x_pos )
                    # Get the Hilbert transform
                    Lambda = hybridisation.get_Lambda_on_grid()
                    Lambda = normalise_na_quantities( Lambda, x_pos )
                    # Get the line representing the eps - eps_a state
                    eps_a_line = hybridisation.get_energy_diff_on_grid() 
                    eps_a_line = normalise_na_quantities( eps_a_line, x_pos )

                    if a == 0:
                        ax[1,j].plot(hybridisation.eps, Delta, color='tab:grey')
                        ax[1,j].fill_between(hybridisation.eps, x_pos, Delta, color='tab:grey', alpha=0.2)

                    # Get the adsorbate density of states
                    dos = hybridisation.get_dos_on_grid()
                    dos = normalise_na_quantities( dos, x_pos )

                    if a in [0, NUMBER_OF_ADSORBATES-1]: 
                        ax[1,j].plot(hybridisation.eps, dos, color=color[a], )
                        ax[1,j].fill_between(hybridisation.eps[occupied_energy_index],
                                             x_pos*np.ones(len(occupied_energy_index)), 
                                             dos[occupied_energy_index], color=color[a], alpha=0.25)

            # Now plot the energies for each row
            jna = NorskovNewnsAnderson(
                Vsd = parameters_metal['Vsd'],
                eps_a = eps_a,
                width = parameters_metal['width'],
                eps = EPS_RANGE, 
                eps_sp_max=EPS_SP_MAX,
                eps_sp_min=EPS_SP_MIN,
                Delta0_mag=delta0,
            )

            # Gather energies
            total_energy = jna.fit_parameters( [alpha, beta, constant], eps_d_range)
            hyb_spd_energy = jna.spd_hybridisation_energy
            # This energy will be used as a basis for comparison
            energy_to_plot = hyb_spd_energy
            energy_to_plot = np.array(energy_to_plot)

            # Plot the total energy as a function of the d-band centre.
            total_energy_adsorbate.append(total_energy)

            # The parameter to plot would be the largest eps_d at which 
            # the energy_to_plot reaches Delta0.
            # First find the index at which energy_to_plot becomes delta0
            index_delta0 = np.argwhere(energy_to_plot > -delta0).flatten()
            # Find the maximum value of eps_d at which this occurs
            if len(index_delta0) > 0:
                eps_d_desc = np.max(np.array(parameters_metal['eps_d'])[index_delta0])
            else:
                eps_d_desc = None
            
            # Store the description of the fit
            argmax_derivative.append(eps_d_desc)

            if a in [0, NUMBER_OF_ADSORBATES-1]: 
                if a == 0:
                    index = 0
                else:
                    index = 1
                    ax[0,0][index].set_xlabel('$\epsilon_d$ (eV)')

                ax[0,0][index].plot(parameters_metal['eps_d'], 
                                    energy_to_plot, 
                                    # alpha=0.75,
                                    ls='-',
                                    lw=2,
                                    alpha=0.75,
                                    color=color_row[j])
                ax[0,0][index].set_ylabel('$E_{\mathregular{hyb}}$ (eV)')
                ax[0,0][index].annotate('$\epsilon_a = %1.1f$ eV'%eps_a,
                                        xy=(0.05, 0.05),
                                        xycoords='axes fraction',
                                        fontsize=12,
                                        )

        # Plot the maximum derivative as a function of eps_a for the different metal rows
        ax[0,2].plot(eps_a_range, argmax_derivative, '-',
                     color=color_row[j], lw=2, label=f'{j+3}d', alpha=0.75)

        # Store the energies to plot against each other in the form of scaling
        final_energy_scaling[j]['min_energy'].extend(np.min(total_energy_adsorbate, axis=0).tolist())
        final_energy_scaling[j]['max_energy'].extend(np.max(total_energy_adsorbate, axis=0).tolist())

    # Plot the scaling line
    for row_index, metal_row in enumerate(final_energy_scaling):
        ax[0,1].plot(final_energy_scaling[metal_row]['max_energy'], 
                     final_energy_scaling[metal_row]['min_energy'], 
                     color=color_row[metal_row], alpha=0.75, 
                     ls='-', label=f'Row: {metal_row+3}', lw=2)
    ax[0,1].set_xlabel(r'$\Delta E_{\mathregular{C}}$ (eV)')
    ax[0,1].set_ylabel(r'$\Delta E_{\mathregular{O}}$ (eV)')

    ax[0,2].legend(loc='best', fontsize=14)
    ax[0,2].set_xlabel(r'$\epsilon_{a}$ (eV)')
    ax[0,2].set_ylabel(r'$\epsilon_s$ (eV)')
    alphabet = list(string.ascii_lowercase)
    i = 0
    for index, a in enumerate(ax.flatten()):
        try:
            a.annotate(alphabet[i]+')', xy=(0.01, 0.6), xycoords='axes fraction')
            i += 1
        except AttributeError:
            a[0].annotate(alphabet[i]+')', xy=(0.01, 0.6), xycoords='axes fraction')
            a[1].annotate(alphabet[i+1]+')', xy=(0.01, 0.6), xycoords='axes fraction')
            i += 2

    fig.savefig(f'output/figure_4_{COMP_SETUP}.png', dpi=300)
    