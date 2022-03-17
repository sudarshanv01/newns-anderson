"""Compare the types of scaling with the same metal and with different metals."""

import numpy as np
import matplotlib.pyplot as plt
from catchemi import ( NewnsAndersonLinearRepulsion,
                       NewnsAndersonNumerical,
                       NewnsAndersonDerivativeEpsd,
                       FitParametersNewnsAnderson
                     )  
import pickle
from create_coupling_elements import create_coupling_elements
from plot_params import get_plot_params
import yaml
import json
from collections import defaultdict
from fitting_functions import get_fitted_function
from plot_figure_4 import set_same_limits
get_plot_params()
from adjustText import adjust_text
import string
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl']

def figure_layout():
    """Get the figure layout for the comparison plot."""
    fig, ax = plt.subplots(2, 3, figsize=(6.75, 5), constrained_layout=True)
    for a in ax[0]:
        a.set_ylabel(r'$E_{\rm O}$ (eV)')
        a.set_xlabel(r'$E_{\rm C}$ (eV)')
    for a in ax[1]:
        a.set_ylabel(r'$E_{\mathrm{hyb}^\prime}$  (eV)')
        a.set_xlabel(r'$\epsilon_d$ (eV)')

    return fig, ax[0,:], ax[1,:]

def aux_fig_layout():
    """Get the layout for the auxilliary figures."""
    fig, ax = plt.subplots(2, 3, figsize=(6.75, 5), constrained_layout=True)
    for i, axp in enumerate(ax):
        for a in axp:
            a.set_ylabel(r'$E_{\rm %s}$ (eV)'%ADSORBATES[i])
            a.set_xlabel(r'$\epsilon_d$ (eV)')
    return fig, ax

if __name__ == '__main__':
    """Compare scaling between different rows of transition
    metals and then on top of that plot the scaling assuming
    that the only one metal is used to carry out scaling. This
    test will allow us to determine if there is a different
    scaling line between the fitted line of a series of transition 
    metals vs. just one transition metal, with different facets."""

    # The chosen computational setup.  
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = open('chosen_setup', 'r').read() 
    # Remove these metals from consideration because they are not part of the study.
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        c_parameters = json.load(f)
    adsorbate_params = {'O': o_parameters, 'C': c_parameters}
    # Read in the metal fitting splines
    with open(f"output/spline_objects.pkl", 'rb') as f:
        spline_objects = pickle.load(f)

    GRID_LEVEL = 'high' # 'high' or 'low'

    # Choose a sequence of adsorbates
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1 ] # eV
    EPS_VALUES = np.linspace(-20, 20, 1000)
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CONSTANT_DELTA0 = 0.1
    if GRID_LEVEL == 'high':
        GRID_SIZE = 120
    elif GRID_LEVEL == 'low':
        GRID_SIZE = 10

    # Figure layout for comparison.
    fig, ax, axd = figure_layout()
    figa, axa = aux_fig_layout()

    # Input parameters to help with the dos from Newns-Anderson
    data_from_dos_calculation = json.load(open(f"output/pdos_moments_{COMP_SETUP['dos']}.json")) 
    data_from_energy_calculation = json.load(open(f"output/adsorption_energies_{COMP_SETUP[CHOSEN_SETUP]}.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    no_of_bonds = yaml.safe_load(open('inputs/number_bonds.yaml', 'r'))
    dft_Vsdsq = json.load(open(f"output/dft_Vsdsq.json"))
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    minmax_parameters = json.load(open('output/minmax_parameters.json'))

    for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        # Create a plot of the chemisorption energy as a function
        # of the epsilon_d for the metals used in this study.

        # Store the parameters in order of metals in this list
        parameters = defaultdict(list)

        # Consider only a specific range of metals in the analysis
        # Those used in Figures 1-3 of the paper
        filling_min, filling_max = minmax_parameters[str(j)]['filling']
        eps_d_min, eps_d_max = minmax_parameters[str(j)]['eps_d']
        # Create linearlised variation of this parameter for
        # the chosen set of materials
        filling_range = np.linspace(filling_max, filling_min, GRID_SIZE, endpoint=True)
        eps_d_range = np.linspace(eps_d_min, eps_d_max, GRID_SIZE, endpoint=True)

        s_fit = spline_objects[j]['s']
        Delta_anderson_fit = spline_objects[j]['Delta_anderson']
        wd_fit = spline_objects[j]['width']
        eps_d_fit = spline_objects[j]['eps_d']

        for i, metal in enumerate(metal_row): 
            # Remove metals that are not used in this study.
            if metal in REMOVE_LIST:
                continue
            # Use the d-band center as the x-axis throughout
            d_band_centre = data_from_dos_calculation[metal]['d_band_centre']

            # Pick the d_band_centre closest to the one in eps_d_range
            index_d_band_interp = np.argmin(np.abs(d_band_centre - eps_d_range))
            d_band_centre = eps_d_range[index_d_band_interp]
            parameters['d_band_centre'].append(d_band_centre)
            width = data_from_dos_calculation[metal]['width']
            parameters['width'].append(width)
            Vsd = np.sqrt(dft_Vsdsq[metal])
            parameters['Vsd'].append(Vsd)
            parameters['metal'].append(metal)
            parameters['no_of_bonds'].append(no_of_bonds[CHOSEN_SETUP][metal])

        # Populate the interpolated Vsd and width values
        for l, eps_d_interp in enumerate(eps_d_range):
            # Continuous setting of parameters for each 
            # continous variation of the metal
            filling_fit = spline_objects[j]['filling'](eps_d_interp)
            width = spline_objects[j]['width'](filling_fit) 
            Vsdsq = create_coupling_elements(s_metal=s_fit(filling_fit),
                                            s_Cu=s_data['Cu'],
                                            anderson_band_width=Delta_anderson_fit(filling_fit),
                                            anderson_band_width_Cu=anderson_band_width_data['Cu'],
                                            r=s_fit(filling_fit),
                                            r_Cu=s_data['Cu'],
                                            normalise_by_Cu=True,
                                            normalise_bond_length=True
                                            )
            Vsd = np.sqrt(Vsdsq)
            parameters['Vsd_interp'].append(Vsd)
            parameters['width_interp'].append(width)
            parameters['no_of_bond_interp'].append(no_of_bonds[CHOSEN_SETUP]['average'])

        # Compute separately the chemisorption energy for each row
        # for each adsorbate.
        total_energy = {}
        range_energy = {}
        metal_energy = defaultdict(list)
        analytical_derivative = {}
        metal_derivative = defaultdict(list)
        for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
            # Now plot the energies for each row
            kwargs_fitting = dict(
                Vsd = parameters['Vsd'],
                eps_a = eps_a,
                width = parameters['width'],
                eps = EPS_VALUES, 
                eps_sp_max=EPS_SP_MAX,
                eps_sp_min=EPS_SP_MIN,
                Delta0_mag=CONSTANT_DELTA0,
                store_hyb_energies = True,
                no_of_bonds = parameters['no_of_bonds'],
            )
            fitting_function = FitParametersNewnsAnderson(**kwargs_fitting)
            
            alpha = adsorbate_params[adsorbate]['alpha']
            beta = adsorbate_params[adsorbate]['beta']
            constant = adsorbate_params[adsorbate]['constant_offset']

            # Gather energies
            total_energy[adsorbate] = fitting_function.fit_parameters( [alpha, beta, constant],
                                                                        parameters['d_band_centre'])

            fitting_function.Vsd = parameters['Vsd_interp']
            fitting_function.width = parameters['width_interp']
            fitting_function.no_of_bonds = parameters['no_of_bond_interp']
            range_energy[adsorbate] = fitting_function.fit_parameters( [alpha, beta, constant],
                                                                        eps_d_range )
            # Gather the derivative of the hybridisation energy with eps_d
            f_Vsd = lambda x: spline_objects[j]['Vsd']( spline_objects[j]['filling'](x)  ) 
            f_Vsd_p = lambda x: spline_objects[j]['Vsd'].derivative()( spline_objects[j]['filling'](x)  ) 
            f_wd = lambda x: spline_objects[j]['width']( spline_objects[j]['filling'](x)  ) 
            f_wd_p = lambda x: spline_objects[j]['width'].derivative()( spline_objects[j]['filling'](x)  ) 

            # Get the derivative of the hybridisation energy with eps_d
            derivative = NewnsAndersonDerivativeEpsd(f_Vsd=f_Vsd,f_Vsd_p=f_Vsd_p,
                                                    eps_a=eps_a, eps=EPS_VALUES,
                                                    f_wd=f_wd, f_wd_p=f_wd_p,
                                                    diff_grid=eps_d_range,
                                                    alpha=alpha, beta=beta,
                                                    Delta0_mag=CONSTANT_DELTA0,
                                                    constant_offset=constant)
            analytical_derivative[adsorbate] = derivative.get_hybridisation_energy_prime_epsd()
            
            # Generate lines where the coupling element is now fixed 
            # for a chosen metal value
            for l, vsd_fixed in enumerate(parameters['Vsd'][:-1]):
                fitting_function.Vsd = vsd_fixed * np.ones(len(eps_d_range))
                fitting_function.width = parameters['width'][l] * np.ones(len(eps_d_range))                 
                energy_fixed = fitting_function.fit_parameters( [alpha, beta, constant], eps_d_range)
                metal_energy[adsorbate].append(energy_fixed)
                # Same for the derivative, here all the coupling elements and width are fixed
                f_Vsd = lambda x: vsd_fixed
                f_Vsd_p = lambda x: 0
                f_wd = lambda x: parameters['width'][l]
                f_wd_p = lambda x: 0
                derivative.f_Vsd = f_Vsd
                derivative.f_Vsd_p = f_Vsd_p
                derivative.f_wd = f_wd
                derivative.f_wd_p = f_wd_p
                metal_derivative[adsorbate].append(derivative.get_hybridisation_energy_prime_epsd())
            
            # Plot the adsorbate energy vs. eps_d
            axa[i,j].plot(parameters['d_band_centre'], total_energy[adsorbate], 'o', color='k')
            axa[i,j].plot(eps_d_range, range_energy[ADSORBATES[i]], '-', color='tab:blue')
            for l, energy in enumerate(metal_energy[ADSORBATES[i]]):
                axa[i,j].plot(eps_d_range, energy, '-', color='tab:orange', alpha=0.5)

        ax[j].plot(total_energy['C'], total_energy['O'], 'o', color='k')
        ax[j].plot(range_energy['C'], range_energy['O'], '-', color='tab:blue', label='Across row scaling')
        axd[j].plot(eps_d_range, analytical_derivative['O'], '-', color='tab:red', label='Across row (O)')
        axd[j].plot(eps_d_range, analytical_derivative['C'], '-', color='tab:green', label='Across row (C)')

        text = []
        text_aux = defaultdict(list)
        for m, metal in enumerate(parameters['metal']):
            text.append(ax[j].text(total_energy['C'][i], total_energy['O'][i], metal, ))
            text_aux['O'].append(axa[0,j].text(parameters['d_band_centre'][m], total_energy['O'][m], metal, ))
            text_aux['C'].append(axa[1,j].text(parameters['d_band_centre'][m], total_energy['C'][m], metal, ))
        adjust_text(text, ax=ax[j])
        adjust_text(text_aux['O'], ax=axa[0,j])
        adjust_text(text_aux['C'], ax=axa[1,j])

        # Plot the individual metal lines
        for l in range(len(metal_energy['O'])):
            if l == 0:
                ax[j].plot(metal_energy['C'][l], metal_energy['O'][l], '-', color='tab:orange',
                            alpha=0.5, label='Single metal')
                axd[j].plot(eps_d_range, metal_derivative['O'][l], '--', color='tab:red', alpha=0.5,
                                label='Single metal (O)')
                axd[j].plot(eps_d_range, metal_derivative['C'][l], '--', color='tab:green', alpha=0.5,
                                label='Single metal (C)')
            else:
                ax[j].plot(metal_energy['C'][l], metal_energy['O'][l], '-', color='tab:orange',
                            alpha=0.5)
                # Plot the derivative
                axd[j].plot(eps_d_range, metal_derivative['O'][l], '--', color='tab:red', alpha=0.5)
                axd[j].plot(eps_d_range, metal_derivative['C'][l], '--', color='tab:green', alpha=0.5)

        ax[j].set_title(f'{j+3}d transition metals')

    ax[1].legend(loc='lower right',fontsize=6)
    axd[2].legend(loc='lower right',fontsize=6)
    set_same_limits(axes=ax)
    set_same_limits(axes=axa[0,:])
    set_same_limits(axes=axa[1,:])
    alphabet = list(string.ascii_lowercase)
    for i, a in enumerate(list(ax) + list(axd)):
        if i in [0, 1, 2]:
            a.annotate(alphabet[i]+')', xy=(0.1, 0.8), fontsize=8, xycoords='axes fraction')
        else:
            a.annotate(alphabet[i]+')', xy=(0.1, 0.1), fontsize=8, xycoords='axes fraction')
    
    for i, a in enumerate(list(axa.flatten())):
        a.annotate(alphabet[i]+')', xy=(0.01, 1.05), xycoords='axes fraction')

    # axa[0,1].annotate('$V_{ak}^2=\mathrm{const}, w_d=\mathrm{const}$', xy=(0.05, 0.1), xycoords='axes fraction',
    #                     color='tab:orange', fontsize=8)
    # axa[1,1].annotate('$V_{ak}^2 \propto \epsilon_d^{-1},w_d\propto \epsilon_d^2$', xy=(0.05, 0.1), xycoords='axes fraction',
    #                     color='tab:blue',fontsize=8)

    fig.savefig('output/compare_types_scaling.png', dpi=300)

    figa.savefig('output/compare_types_scaling_aux_plot.png', dpi=300)