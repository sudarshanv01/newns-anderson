"""Plot an idealised scaling model for C and O scaling."""
import numpy as np
import scipy
import json
from NewnsAnderson import JensNewnsAnderson, NewnsAndersonNumerical
from collections import defaultdict
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from plot_fitting_metal_parameters import func_a_by_r, func_a_r2
from scipy.optimize import curve_fit
from scipy import special
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

def create_plot_layout():
    """Create a plot layout for plotting the Newns-Anderson
    dos and the energies of orthogonalisation, spd hybridisation
    energy for each specific adsorbate."""
    fig = plt.figure(figsize=(14,10), constrained_layout=True)
    gs = fig.add_gridspec(nrows=14, ncols=3,)
    # The first 2 rows will be the orthogonalisation energies, spd
    # hybridisation energy and the total energy as a function of the
    # d-band centre.
    ax1 = fig.add_subplot(gs[0:4, 0])
    ax2 = fig.add_subplot(gs[0:4, 1])
    ax3 = fig.add_subplot(gs[0:4, 2])
    # # Get the na graph
    # ax1o = fig.add_subplot(gs[3:5, 0])
    # ax2o = fig.add_subplot(gs[3:5, 1])
    # ax3o = fig.add_subplot(gs[3:5, 2])
    # ax1o.set_ylabel('$n_a$ (e)')
    
    # Set the axes labels
    ax1.set_title('3d')
    ax2.set_title('4d')
    ax3.set_title('5d')
    ax1.set_xlabel('$\epsilon_{d} - \epsilon_{F}$')
    ax2.set_xlabel('$\epsilon_{d} - \epsilon_{F}$')
    ax3.set_xlabel('$\epsilon_{d} - \epsilon_{F}$')
    # Each of ax1, ax2 and ax3 have a twinx() axis, which is used to
    # plot the derivative
    ax1.set_ylabel('$\Delta E_{\mathregular{hyb}}$ / eV (--)')
    # Then make three plots with the density of states coming from
    # the different solutions of the Newns-Anderson equation.
    ax4 = fig.add_subplot(gs[4:, 0])
    ax5 = fig.add_subplot(gs[4:, 1])
    ax6 = fig.add_subplot(gs[4:, 2])
    # Set the axes labels
    ax4.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax5.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax6.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax4.set_ylabel('Projected Density of States')
    # Remove y-ticks from 4, 5, 6
    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])

    return fig, np.array([ [ax1, ax2, ax3], [ax4, ax5, ax6] ]) # [ax1o, ax2o, ax3o]

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

def functional_form_energy(eps_d, a, b, c):
    """Fitting function for the functional form of the energy."""
    eps_d = np.array(eps_d)
    cutoff_function = lambda x:  ( special.erfc(eps_d - b) + 1 ) / 2 # np.exp(np.abs(b) * eps_d) 
    return a * eps_d * cutoff_function(eps_d) + c 

if __name__ == '__main__':
    """Plot the hybridisation energy as a function of the d-band
    centre for 3d, 4d and 5d metals. Change eps_a to see when the curve
    levels off. Assume that the fractional filling depends linearly on
    the d-band centre. This assumption is used so that we can get smooth 
    variations in the figure."""

    FUNCTIONAL = 'PBE_scf'
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{FUNCTIONAL}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{FUNCTIONAL}.json", 'r') as f:
        c_parameters = json.load(f)
    with open(f"output/fitting_metal_parameters_{FUNCTIONAL}.json", 'r') as f:
        metal_parameters = json.load(f)

    # Create range of parameters 
    NUMBER_OF_ADSORBATES = 10
    NUMBER_OF_METALS = 30
    PLOT_METAL_DOS = 6

    eps_a_range = create_range_parameters(o_parameters['eps_a'], c_parameters['eps_a'], n=NUMBER_OF_ADSORBATES)
    alpha_range = create_range_parameters(o_parameters['alpha'], c_parameters['alpha'], n=NUMBER_OF_ADSORBATES)
    beta_range  = create_range_parameters(o_parameters['beta'], c_parameters['beta'], n=NUMBER_OF_ADSORBATES)
    delta0_range = create_range_parameters(o_parameters['delta0'], c_parameters['delta0'], n=NUMBER_OF_ADSORBATES)

    # Input parameters to help with the dos from Newns-Anderson
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{FUNCTIONAL}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{FUNCTIONAL}.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Each column is for a different row of transition metals
    fig, ax = create_plot_layout() 
    fige, axe = create_aux_plot_layout()

    # get a color cycle for the different adsorbates based on viridis
    color = plt.cm.viridis(np.linspace(0, 1, NUMBER_OF_ADSORBATES))
    color_row = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

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
            # Store the metal parameters to be used in the JNA model
            parameters_metal = defaultdict(list)
            for i, filling in enumerate(filling_range):

                Vsd = np.sqrt(func_a_by_r( filling, *Vsdsq_fit ) )
                eps_d = eps_d_range[i]                
                width = func_a_r2( filling, *wd_fit )
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
                    eps = np.linspace(-15, 10, 1000),
                    k = delta0,
                )
                hybridisation.calculate_energy()
            
                # Decide on the x-position based on the d-band centre
                denote_pos = NUMBER_OF_METALS / PLOT_METAL_DOS

                # Get the metal projected density of states
                # Pick every PLOT_METAL_DOS number of points
                if i % PLOT_METAL_DOS == 0:
                    x_pos =  2 * i / denote_pos
                    Delta = normalise_na_quantities( hybridisation.Delta, x_pos )
                    # Get the hilbert transform
                    Lambda = normalise_na_quantities( hybridisation.Lambda, x_pos )
                    # Get the line representing the eps - eps_a state
                    eps_a_line = hybridisation.eps - hybridisation.eps_a
                    eps_a_line = normalise_na_quantities( eps_a_line, x_pos )
                    if a == 0:
                        ax[1,j].plot(hybridisation.eps, Delta, color='tab:red', lw=3)
                        ax[1,j].fill_between(hybridisation.eps, x_pos*np.ones(len(Delta)), Delta, color='tab:red', alpha=0.10)
                        ax[1,j].plot(hybridisation.eps, Lambda, color='tab:blue', lw=3, alpha=0.5)

                    # Get the adsorbate density of states
                    na = hybridisation.dos 
                    na = normalise_na_quantities( na, x_pos )

                    if a in [0, NUMBER_OF_ADSORBATES-1]: 
                        ax[1,j].plot(hybridisation.eps, na, color=color[a], lw=3)
                        ax[1,j].fill_between(hybridisation.eps, x_pos*np.ones(len(na)), na, color=color[a], alpha=0.10)
                        # axo[j].plot(eps_d, hybridisation.na, 'o', color=color[a])

            # Now plot the energies for each row
            jna = JensNewnsAnderson(
                Vsd = parameters_metal['Vsd'],
                eps_a = eps_a,
                width = parameters_metal['width'],
                filling = parameters_metal['filling'],
            )

            total_energy = jna.fit_parameters(eps_ds = eps_d_range, alpha=alpha, beta=beta, constant=delta0)
            spd_hyb_energy = jna.spd_hybridisation_energy
            energy_to_plot =  total_energy 
            energy_to_plot = np.array(energy_to_plot)

            # Plot the hybridisation energy as a function of the d-band centre.
            total_energy_adsorbate.append(energy_to_plot)
            # Fit the points with the functional form of the hybridisation energy
            # popt, pcov = curve_fit(functional_form_energy, parameters_metal['eps_d'], energy_to_plot)
            # Get the polynomial coefficients
            coeffs = np.polyfit(parameters_metal['eps_d'], energy_to_plot, 3)
            # get the derivative coefficients
            # coeffs_derivative = np.polyder(coeffs, 1)
            # Find the parameter to plot as a sign of the decay of the energy
            # argmax_derivative.append(popt[1])
            # Get the argmax of the coeff derivaitive
            # max_derivative = np.argmax(np.polyval(coeffs_derivative, parameters_metal['eps_d']))
            # argmax_derivative.append(parameters_metal['eps_d'][max_derivative])
            argmax_derivative.append(coeffs[0])
            if a in [0, NUMBER_OF_ADSORBATES-1]: 
                ax[0, j].plot(parameters_metal['eps_d'], energy_to_plot, 'o', color=color[a])
                # ax[0, j].plot(parameters_metal['eps_d'], functional_form_energy(parameters_metal['eps_d'], *popt), '--', color=color[a], label=f'$\epsilon_a$ = {eps_a:.1f} eV')
                # plot the polynomial fit
                ax[0, j].plot(parameters_metal['eps_d'], np.polyval(coeffs, parameters_metal['eps_d']), '-', color=color[a], label=f'$\epsilon_a$ = {eps_a:.1f} eV')
                # plot the derivative fit
                # ax2[j].plot(parameters_metal['eps_d'], np.polyval(coeffs_derivative, parameters_metal['eps_d']), '--', color=color[a], label=f'$\epsilon_a$ = {eps_a:.1f} eV')
                # Plot the maximum of the derivative
                # ax2[j].plot(parameters_metal['eps_d'][max_derivative], np.polyval(coeffs_derivative, parameters_metal['eps_d'][max_derivative]), 'o', color=color[a])

        # Plot a fill_between between the highest and lowest total energy
        ax[0, j].fill_between(eps_d_range, 
                              np.min(total_energy_adsorbate, axis=0), 
                              np.max(total_energy_adsorbate, axis=0),
                              color='tab:gray', alpha=0.25)
        # Plot the maximum derivative as a function of eps_a for the different metal rows
        axe.plot(eps_a_range, argmax_derivative, 'o-', color=color_row[j], label=f'Row: {j+1}')
        # Add a axhline for Pt eps_d
    # ax[0, 2].axvline(-2.24, color='k', linestyle='--',)
    # Annotate that the line is Pt
    # ax[0, 2].text(-2.24, -4, 'Pt', fontsize=12)

    # ax[0,1].axvline(-1.75, color='k', linestyle='--',)
    # ax[0,1].text(-1.75, -4, 'Pd', fontsize=12)

    set_same_limits(ax[0,:])
    ax[0,0].legend(loc='best', fontsize=14)
    axe.legend(loc='best', fontsize=14)
    fig.savefig('output/energy_metal_row.png', dpi=300)
    fige.savefig('output/derivative_descriptor.png', dpi=300)
        
    