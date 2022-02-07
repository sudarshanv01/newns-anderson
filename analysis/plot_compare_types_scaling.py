"""Compare the types of scaling with the same metal and with different metals."""

import numpy as np
import matplotlib.pyplot as plt
from norskov_newns_anderson.NewnsAnderson import NorskovNewnsAnderson, NewnsAndersonNumerical
from plot_params import get_plot_params
import yaml
import json
from plot_figure_4 import func_a_by_r, func_a_r_sq, set_same_limits
from collections import defaultdict
get_plot_params()
from adjustText import adjust_text

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl']

def figure_layout():
    """Get the figure layout for the comparison plot."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for a in ax:
        a.set_ylabel(r'$\Delta E_{\rm O}$ (eV)')
        a.set_xlabel(r'$\Delta E_{\rm C}$ (eV)')

    return fig, ax

if __name__ == '__main__':
    """Compare scaling between different rows of transition
    metals and then on top of that plot the scaling assuming
    that the only one metal is used to carry out scaling. This
    test will allow us to determine if there is a different
    scaling line between the fitted line of a series of transition 
    metals vs. just one transition metal, with different facets."""

    # The chosen computational setup.  
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))['group'][0]
    # Remove these metals from consideration because they are not part of the study.
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{COMP_SETUP}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{COMP_SETUP}.json", 'r') as f:
        c_parameters = json.load(f)
    adsorbate_parameters = {'C': c_parameters, 'O': o_parameters}
    with open(f"output/fitting_metal_parameters_{COMP_SETUP}.json", 'r') as f:
        metal_parameters = json.load(f)

    # Choose a sequence of adsorbates
    ADSORBATES = ['O', 'C']
    EPS_A_VALUES = [ -5, -1 ] # eV
    EPS_VALUES = np.linspace(-20, 20, 1000)
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CONSTANT_DELTA0 = 0.1
    GRID_SIZE = 120

    # Figure layout for comparison.
    fig, ax = figure_layout()

    # Input parameters to help with the dos from Newns-Anderson
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{COMP_SETUP}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{COMP_SETUP}.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
        # Create a plot of the chemisorption energy as a function
        # of the epsilon_d for the metals used in this study.

        # Store the parameters in order of metals in this list
        parameters = defaultdict(list)

        # Get parameters of continuous fitting for each row
        Vsdsq_fit = metal_parameters['Vsdsq'][str(j)]
        wd_fit = metal_parameters['width'][str(j)]
        # Consider only a specific range of metals in the analysis
        # Those used in Figures 1-3 of the paper
        filling_min, filling_max = metal_parameters['filling_minmax'][str(j)]
        eps_d_min, eps_d_max = metal_parameters['eps_d_minmax'][str(j)]
        # Create linearlised variation of this parameter for
        # the chosen set of materials
        filling_range = np.linspace(filling_max, filling_min, GRID_SIZE, endpoint=True)
        eps_d_range = np.linspace(eps_d_min, eps_d_max, GRID_SIZE, endpoint=True)
        # Truncated eps_d range based on the expected metal-only scaling plot
        # eps_d_range_trunc = np.linspace()
        parameters['Vsd_interp'] = np.sqrt( func_a_by_r( filling_range, *Vsdsq_fit ) )
        parameters['width_interp'] = func_a_r_sq(filling_range, *wd_fit)

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

            width = parameters['width_interp'][index_d_band_interp] 
            parameters['width'].append(width)

            Vsd = parameters['Vsd_interp'][index_d_band_interp]
            parameters['Vsd'].append(Vsd)

            parameters['metal'].append(metal)
        
        # Compute separately the chemisorption energy for each row
        # for each adsorbate.
        total_energy = {}
        range_energy = {}
        metal_energy = defaultdict(list)
        for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
            fitting_function =  NorskovNewnsAnderson(
                Vsd = parameters['Vsd'],
                width = parameters['width'],
                eps_a = eps_a,
                eps_sp_min = EPS_SP_MIN,
                eps_sp_max = EPS_SP_MAX,
                eps = EPS_VALUES,
                Delta0_mag = CONSTANT_DELTA0)
            
            alpha = adsorbate_parameters[adsorbate]['alpha']
            beta = adsorbate_parameters[adsorbate]['beta']
            constant = adsorbate_parameters[adsorbate]['constant_offset']

            # Gather energies
            total_energy[adsorbate] = fitting_function.fit_parameters( [alpha, beta, constant],
                                                                        parameters['d_band_centre'])
            fitting_function.Vsd = parameters['Vsd_interp']
            fitting_function.width = parameters['width_interp']
            range_energy[adsorbate] = fitting_function.fit_parameters( [alpha, beta, constant],
                                                                        eps_d_range )
            
            # Generate lines where the coupling element is now fixed 
            # for a chosen metal value
            for l, vsd_fixed in enumerate(parameters['Vsd'][:-1]):
                fitting_function.Vsd = vsd_fixed * np.ones(len(eps_d_range))
                fitting_function.width = parameters['width'][l] * np.ones(len(eps_d_range))                 
                energy_fixed = fitting_function.fit_parameters( [alpha, beta, constant], eps_d_range)
                metal_energy[adsorbate].append(energy_fixed)

        ax[j].plot(total_energy['C'], total_energy['O'], 'o', color='k')
        ax[j].plot(range_energy['C'], range_energy['O'], '-', color='tab:blue', label='Across row scaling')
        text = []
        for i, metal in enumerate(parameters['metal']):
            text.append(ax[j].text(total_energy['C'][i], total_energy['O'][i], metal, fontsize=12))
        adjust_text(text, ax=ax[j])

        # Plot the individual metal lines
        for l in range(len(metal_energy['O'])):
            if l == 0:
                ax[j].plot(metal_energy['C'][l], metal_energy['O'][l], '-', color='tab:red',
                            alpha=0.5, label='Single metal scaling')
            else:
                ax[j].plot(metal_energy['C'][l], metal_energy['O'][l], '-', color='tab:red',
                            alpha=0.5)

        ax[j].set_title(f'{j+3}d transition metals')

    ax[1].legend(loc='lower right', fontsize=11)
    set_same_limits(axes=ax)
    
    fig.savefig('output/compare_types_scaling.png', dpi=300)