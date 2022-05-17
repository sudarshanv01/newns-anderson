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
C_COLOR = 'tab:purple'
O_COLOR = 'tab:orange'

def create_plot_layout():
    """Create a plot layout for plotting the Newns-Anderson
    dos and the energies of orthogonalisation, spd hybridisation
    energy for each specific adsorbate."""
    # Create 3x3 grid of plots
    fig, ax = plt.subplots(3, 3, figsize=(6.9, 5), constrained_layout=True)

    # Twin x-axis for the first column
    ax_p = [a.twinx() for a in ax[:-1, 0]]
    ax_p = np.array(ax_p).reshape(-1)

    # Twin x-axis for the second column
    ax_n = [a.twinx() for a in ax[:-1, 1]]
    ax_n = np.array(ax_n).reshape(-1)

    for i, a in enumerate(ax[:-1]):
        a[0].set_xlabel(r'$\epsilon_d$ / eV')
        a[1].set_xlabel(r'$\epsilon_d$ / eV')
        if i == 1:
            a[2].set_xlabel(r'$\epsilon_a$ / eV')
        # Make a twin axis for the hybridisation energy panel
        a[0].set_ylabel(r'$E_{\rm hyb}$  %s* / eV'%ADSORBATES[i])
        ax_p[i].set_ylabel(r'$E_{\rm hyb} ^\prime$  %s* / eV'%ADSORBATES[i], color='tab:grey')
        a[1].set_ylabel(r'$E_{\mathrm{ortho}}$  %s* / eV'%ADSORBATES[i])
        ax_n[i].set_ylabel(r'$\left ( n_a + f \right )$  %s* / e'%ADSORBATES[i], color='tab:grey')

        # Have an arrow pointing to the two y-axis showing the right marker
        a[0].annotate('$--$', xy=(1,0.95), xytext=(0.8, 0.95), xycoords='axes fraction',
                    arrowprops=dict(facecolor='black', width=0.1, headwidth=2.5),
                    horizontalalignment='right', verticalalignment='center', color='tab:grey')
        a[1].annotate('$--$', xy=(1,0.95), xytext=(0.8, 0.95), xycoords='axes fraction',
                    arrowprops=dict(facecolor='black', width=0.1, headwidth=2.5, ),
                    horizontalalignment='right', verticalalignment='center', color='tab:grey')
        a[0].annotate('$-$', xy=(0,0.1), xytext=(0.25, 0.1), xycoords='axes fraction',
                    arrowprops=dict(facecolor='black', width=0.1, headwidth=2.5),
                    horizontalalignment='right', verticalalignment='center',)
        a[1].annotate('$-$', xy=(0,0.1), xytext=(0.25, 0.1), xycoords='axes fraction',
                    arrowprops=dict(facecolor='black', width=0.1, headwidth=2.5),
                    horizontalalignment='right', verticalalignment='center',)
        ax_n[i].tick_params(axis='y', labelcolor='tab:grey')
        ax_p[i].tick_params(axis='y', labelcolor='tab:grey')

    ax[-1,0].set_ylabel(r'$E_{\mathrm{hyb}}$ %s* / eV'%ADSORBATES[0])
    ax[-1,1].set_ylabel(r'$E_{\mathrm{ortho}}$ %s* / eV'%ADSORBATES[0])
    ax[-1,0].set_xlabel(r'$E_{\mathrm{hyb}}$ %s* / eV'%ADSORBATES[-1])
    ax[-1,1].set_xlabel(r'$E_{\mathrm{ortho}}$ %s* / eV'%ADSORBATES[-1])
    # ax[-1,-1].set_xlabel(r'$\left ( n_a + f \right )$ %s* / e'%ADSORBATES[-1])
    # ax[-1,-1].set_ylabel(r'$\left ( n_a + f \right )$ %s* / e'%ADSORBATES[0])

    ax[0,-1].set_yticks([])
    ax[0,-1].set_ylabel(r'Projected density of states', fontsize=8)
    ax[0,-1].set_xlabel(r'$\epsilon - \epsilon_F$ / eV')

    # Make a legend on the last plot to show 
    # the different eps_d ranges
    # for i in [0, 1]:
    #     ax[-1,i].plot([], [], color='tab:grey', label='Saturated $\epsilon_d$')
    #     ax[-1,i].plot([], [], color='k', label='$\epsilon_d$ > $\epsilon_s$')
    #     ax[-1,i].legend(loc='lower right', frameon=False, fontsize=6)
        # ax[-1,i].legend(bbox_to_anchor=(1.04,1), borderaxespad=0)


    ax[1,2].set_ylabel(r'$\epsilon_s$ (eV)')
    # ax[0,2].axis('off')

    return fig, ax, ax_p, ax_n 

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
    
def create_range_parameters(param1, param2, n=51):
    """Linearly interpolate between two parameters."""
    return np.linspace(param1, param2, n)

def normalise_na_quantities(quantity, x_add=0, per_max=True):
    """Utility function to align the density of states for Newns-Anderson plots."""
    if per_max:
        return quantity / np.max(quantity) + x_add
    else:
        return quantity + x_add

def plot_epsa_value(eps_a):
    """Function to evaluate if eps_a value is to 
    be plotted."""
    chosen_values = CHOSEN_EPS_A_TO_PLOT 
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
    CHOSEN_SETUP = open('chosen_setup', 'r').read() 
    REPULSION = 'linear'
    # Read in scaling parameters from the model.
    with open(f"output/O_repulsion_{REPULSION}_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_repulsion_{REPULSION}_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        c_parameters = json.load(f)
    adsorbate_params = {'O': o_parameters, 'C': c_parameters}
    GRID_LEVEL = 'low' # 'high' or 'low'
    color_ads = [O_COLOR, C_COLOR]

    # Create range of parameters 
    if GRID_LEVEL == 'high':
        NUMBER_OF_ADSORBATES = 21
        NUMBER_OF_METALS = 50
        GRID_SPACING_DERIV = 300
    elif GRID_LEVEL == 'low':
        NUMBER_OF_ADSORBATES = 5
        NUMBER_OF_METALS = 10
        GRID_SPACING_DERIV = 10
    
    # Make sure the number of adsorbate is odd
    assert NUMBER_OF_ADSORBATES % 2 == 1, 'Number of adsorbates must be odd.'

    EPS_RANGE = np.linspace(-30, 20, 1000)
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1 ] # eV
    CHOSEN_EPS_A_TO_PLOT = [ *EPS_A_VALUES, -3]
    CHOSEN_METAL = 'Rh'

    # Fix the energy width of the sp 
    # states of the metal
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CUTOFF_VALUE = 0.1 # Any value lower than this considered saturated

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
    no_of_bonds = yaml.safe_load(open('inputs/number_bonds.yaml', 'r'))
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    minmax_parameters = json.load(open('output/minmax_parameters.json'))

    with open(f"output/spline_objects.pkl", 'rb') as f:
        spline_objects = pickle.load(f)

    # Get the main figure with the energies and the axes
    # and the supporting figure with the density of states
    fig, ax, ax_p, ax_n  = create_plot_layout() 

    # get a color cycle for the different adsorbates based on viridis
    color = plt.cm.coolwarm_r(np.linspace(0, 1, NUMBER_OF_ADSORBATES))
    color_row = ['tab:red', 'tab:blue', 'orange',]
    marker_row = ['o', 's', '^']
    ls_row = ['-', '-', '-']
    ls_eps_a = [':', '-', '-.']

    # Plot the legend for the rows
    for i in range(len(color_row)):
        ax[1,2].plot([], [], color=color_row[i], ls=ls_row[i], label=f'{i+3}' + r'$d$')
    # ax[0,2].legend(loc='center', fontsize=12)
    ax[1,2].legend(bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=8)

    # Store the final energies to plot in a scaling line
    final_energy_scaling = defaultdict( lambda: defaultdict(lambda: defaultdict(list)))

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
            s_fit = spline_objects[j]['s']
            Delta_anderson_fit = spline_objects[j]['Delta_anderson']
            wd_fit = spline_objects[j]['width']
            eps_d_fit = spline_objects[j]['eps_d']

            # Consider only a specific range of metals in the analysis
            # Those used in Figures 1-3 of the paper
            filling_min, filling_max = minmax_parameters[str(j)]['filling']
            eps_d_min, eps_d_max = minmax_parameters[str(j)]['eps_d']
            # Generate grid for differentiation
            diff_grid = np.linspace(eps_d_min, eps_d_max, GRID_SPACING_DERIV)

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
                                                normalise_bond_length=True)
                Vsd = np.sqrt(Vsdsq)

                parameters_metal['Vsd'].append(Vsd)
                parameters_metal['eps_d'].append(eps_d)
                parameters_metal['width'].append(width)
                parameters_metal['filling'].append(filling)
                # TODO: Fix the number of bonds issue
                parameters_metal['no_of_bonds'].append(no_of_bonds[CHOSEN_SETUP]['average'])

            if to_plot: 
                # If the adsorbate is C or O, plot the energy
                # Generate the kwargs to pass on to FitParametersNewnsAnderson
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

                # This energy will be used as a basis for comparison
                # We are looking for the maximum eps_d at which energy_to_plot
                # saturates to its values of -Delta0
                hyb_energy = np.array(jna.hyb_energy)
                ortho_energy = np.array(jna.ortho_energy)
                occupancy = np.array(jna.occupancy)
                filling = np.array(jna.filling_factor)
                na_plus_f = occupancy + filling

                # Store the energies to plot in a scaling line
                final_energy_scaling[eps_a][j]['hyb_energy'].extend(hyb_energy)
                final_energy_scaling[eps_a][j]['ortho_energy'].extend(ortho_energy)
                final_energy_scaling[eps_a][j]['occupancy'].extend(occupancy)
                final_energy_scaling[eps_a][j]['filling'].extend(filling)
                final_energy_scaling[eps_a][j]['na_plus_f'].extend(na_plus_f)
                final_energy_scaling[eps_a][j]['chemisorption_energy'].extend(chemisorption_energy)

                # Plot only if it is an adsorbate we compute
                if eps_a in EPS_A_VALUES:
                    ax[index_plot,0].plot(parameters_metal['eps_d'], 
                                    hyb_energy, '-',
                                    color=color_row[j], ls=ls_row[j],)
                    ax[index_plot,1].plot(parameters_metal['eps_d'],
                                    ortho_energy, '-',
                                    color=color_row[j], ls=ls_row[j],)
                    ax_n[index_plot].plot(parameters_metal['eps_d'],
                                    na_plus_f, '--', alpha=0.4,
                                    color=color_row[j], ls='--',)
                
        
            # Compute the derivative of the hybridisation energy with 
            # the d-band centre to be plotted in the third figure
            # Generate the continuous variation of parameters as a 
            # function of eps_d for each row.
            f_Vsd = lambda x: spline_objects[j]['Vsd']( spline_objects[j]['filling'](x)  ) 
            f_Vsd_p = lambda x: spline_objects[j]['Vsd'].derivative()( spline_objects[j]['filling'](x)  ) 
            f_wd = lambda x: spline_objects[j]['width']( spline_objects[j]['filling'](x)  ) 
            f_wd_p = lambda x: spline_objects[j]['width'].derivative()( spline_objects[j]['filling'](x)  ) 

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
                                if np.abs(analytical_hyb_deriv[a]) < CUTOFF_VALUE]
            # Remove all points that are more than -1 eV
            saturation_epsd_arg = [ a for a in saturation_epsd_arg if diff_grid[a] < -1]
            if len(saturation_epsd_arg) > 0:
                saturation_epsd = diff_grid[saturation_epsd_arg]            
                eps_s[j].append(np.max(saturation_epsd))
                final_energy_scaling[eps_a][j]['eps_s'].extend(saturation_epsd)
                if to_plot and eps_a in EPS_A_VALUES:
                    ax_p[index_plot].plot(np.max(saturation_epsd),
                                          analytical_hyb_deriv[np.argmax(saturation_epsd)],
                                          '*', color=color_row[j],)
            else:
                eps_s[j].append(None)

            if to_plot and eps_a in EPS_A_VALUES:
                # Plot the derivative of the hybridisation energy with eps_d
                ax_p[index_plot].plot(diff_grid, analytical_hyb_deriv,
                                       ls='--', color=color_row[j],
                                       alpha=0.4)
    # Plot the eps_s values for each row
    for i, (row, eps_s_row) in enumerate(eps_s.items()):
        if eps_s_row:
            ax[1,2].plot(eps_a_range, eps_s_row, ls=ls_row[j], color=color_row[i]) 
        
    # Plot the scaling lines between C and O for the hybridisation
    # and orthogonalisation energies
    # for eps_a, row_scaling in final_energy_scaling.items():
    for i, row_index in enumerate(final_energy_scaling[-1.0].keys()):
        ax[-1,0].plot(final_energy_scaling[-1.0][row_index]['hyb_energy'],
                      final_energy_scaling[-5.0][row_index]['hyb_energy'],
                      color=color_row[row_index], ls='-', alpha=0.5)
        # Iterate every 5 values of final_energy_scaling[-1.0][row_index]['hyb_energy']
        for j in range(0, len(final_energy_scaling[-1.0][row_index]['hyb_energy']), 3):
            ax[-1,0].quiver(\
                            final_energy_scaling[-1.0][row_index]['hyb_energy'][j],
                            final_energy_scaling[-5.0][row_index]['hyb_energy'][j],
                            final_energy_scaling[-1.0][row_index]['hyb_energy'][j+1]-\
                            final_energy_scaling[-5.0][row_index]['hyb_energy'][j],
                            final_energy_scaling[-5.0][row_index]['hyb_energy'][j+1]-\
                            final_energy_scaling[-5.0][row_index]['hyb_energy'][j],
                            color = color_row[row_index], 
                            scale_units='xy', scale=5, width=.015,
            ) 
        # Make arrows along the plot in ax[-1,0]

        ax[-1,1].plot(final_energy_scaling[-1.0][row_index]['ortho_energy'],
                      final_energy_scaling[-5.0][row_index]['ortho_energy'],
                      color=color_row[row_index], ls='-',) #alpha=0.2)

    # Run the Newns-Anderson model with the parameters to plot the 
    # projected density of states of the adsorbate and the metal
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Plotting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
        alpha = adsorbate_params[adsorbate]['alpha']
        beta = adsorbate_params[adsorbate]['beta']
        constant_offest = adsorbate_params[adsorbate]['constant_offset']
        CONSTANT_DELTA0 = adsorbate_params[adsorbate]['delta0']
        final_params = [alpha, beta, constant_offest]

        hybridisation = NewnsAndersonNumerical(
            Vak = np.sqrt(beta * dft_Vsdsq[CHOSEN_METAL]),
            eps_a = eps_a, 
            eps_d = data_from_dos_calculation[CHOSEN_METAL]['d_band_centre'],
            width = data_from_dos_calculation[CHOSEN_METAL]['width'],
            eps = EPS_RANGE,
            Delta0_mag = CONSTANT_DELTA0,
            eps_sp_max = EPS_SP_MAX,
            eps_sp_min = EPS_SP_MIN,
        )

        # Get the density of states
        adsorbate_dos = hybridisation.get_dos_on_grid()
        adsorbate_dos = normalise_na_quantities(adsorbate_dos)
        Delta = hybridisation.get_Delta_on_grid()
        Delta = normalise_na_quantities(Delta)

        # plot the adsorbate density of states
        ax[0,-1].plot(EPS_RANGE, adsorbate_dos, color=color_ads[i])
        ax[0,-1].plot(EPS_RANGE, Delta,  color='k', alpha=0.5)
        ax[0,-1].fill_between(EPS_RANGE, adsorbate_dos, color=color_ads[i], alpha=0.5)
        # Annotate the metal name in the top left
        ax[0,-1].text(0.05, 0.95, CHOSEN_METAL, transform=ax[0,-1].transAxes, fontsize=12,
                        verticalalignment='top', horizontalalignment='left')

        occupancy = hybridisation.get_occupancy()
        # Annotate the occupancy on the top right corner of the plot
        ax[0,-1].annotate(r"$n_a$"+f"({adsorbate}*)={occupancy:.2f}", xy=(0.98, 0.8-0.2*i), xycoords='axes fraction',
                        xytext=(-5, 5), textcoords='offset points',
                        ha='right', va='top', color=color_ads[i], fontsize=6,
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
        )
        ax[0,-1].set_xlim(-20, 20)

    # Plot the energies for for each eps_a with the eps_a for O
    for a, eps_a in enumerate(CHOSEN_EPS_A_TO_PLOT):
        for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            # Plot the energies for each row
            ax[-1,-1].plot(final_energy_scaling[-5][j]['ortho_energy'],
                       final_energy_scaling[eps_a][j]['ortho_energy'],
                       color=color_row[j],
                       ls=ls_eps_a[a],
                       label=f"{metal_row}",
                       alpha=0.5)
        # Annotate nearby the line with the eps_a value
        ax[-1,-1].annotate("$\epsilon_a=$"+f"{eps_a} eV", xy=( final_energy_scaling[-5][j]['ortho_energy'][1]+2,
                                            final_energy_scaling[eps_a][j]['ortho_energy'][1]+0.5),
                                xycoords='data',
                                xytext=(-5, 5), textcoords='offset points',
                                ha='right', va='top', color='k', fontsize=6,
                                # bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.25),
            )
    ax[-1,-1].set_xlabel("$E_{\mathrm{ortho}}$ ($-5$ eV) / eV")
    ax[-1,-1].set_ylabel("$E_{\mathrm{ortho}}$ / eV")

    # Add figure numbers
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(ax.T.flatten()):
        a.annotate(alphabet[i]+')', xy=(0.05, 0.5), fontsize=8, xycoords='axes fraction')

    fig.savefig(f'output/figure_4_{COMP_SETUP[CHOSEN_SETUP]}.png', dpi=300)