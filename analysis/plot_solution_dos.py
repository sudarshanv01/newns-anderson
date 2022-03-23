"""Plot the density of states coming from the model."""
import string
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.integrate import simps
from plot_params import get_plot_params
from catchemi import NewnsAndersonNumerical
import yaml
get_plot_params()

C_COLOR = 'tab:purple'
O_COLOR = 'tab:orange'

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',]
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',]
THIRD_ROW   = [ 'X', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',] 

def normalise_na_quantities(quantity, x_add=0, per_max=True):
    """Utility function to align the density of states for Newns-Anderson plots."""
    if per_max:
        return quantity / np.max(quantity) + x_add
    else:
        return quantity + x_add

if __name__ == "__main__":
    """From the solution of the Newns-Anderson model plot the
    density of states."""
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = open('chosen_setup', 'r').read() 

    # Get a cycle of with colormap
    colors =  plt.cm.viridis(np.linspace(0, 1, 10))

    # Get the plot parameters
    fig, ax = plt.subplots(len(FIRST_ROW), 3, figsize=(6.5, 10), constrained_layout=True)

    # Read in the solution
    COMP_SETUP = yaml.safe_load(stream=open('chosen_group.yaml', 'r'))
    CHOSEN_SETUP = open('chosen_setup', 'r').read() 
    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{COMP_SETUP[CHOSEN_SETUP]}.json", 'r') as f:
        c_parameters = json.load(f)
    adsorbate_params = {'O': o_parameters, 'C': c_parameters}

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
    color_ads = [O_COLOR, C_COLOR]
    EPS_A_VALUES = [ -5, -1 ] # eV
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    EPS_VALUES = np.linspace(-30, 20, 1000)

    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Plotting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
        alpha = adsorbate_params[adsorbate]['alpha']
        beta = adsorbate_params[adsorbate]['beta']
        constant_offest = adsorbate_params[adsorbate]['constant_offset']
        CONSTANT_DELTA0 = adsorbate_params[adsorbate]['delta0']
        final_params = [alpha, beta, constant_offest]

        # Make a plot for each metal
        for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            # Iterate over each metal
            for k, metal in enumerate(metal_row):
                if REMOVE_LIST:
                    if metal in REMOVE_LIST:
                        continue
                # Run the Newns-Anderson model
                hybridisation = NewnsAndersonNumerical(
                    Vak = np.sqrt(beta * dft_Vsdsq[metal]),
                    eps_a = eps_a, 
                    eps_d = data_from_dos_calculation[metal]['d_band_centre'],
                    width = data_from_dos_calculation[metal]['width'],
                    eps = EPS_VALUES,
                    Delta0_mag = CONSTANT_DELTA0,
                    eps_sp_max = EPS_SP_MAX,
                    eps_sp_min = EPS_SP_MIN,
                )

                # Get the density of states
                adsorbate_dos = hybridisation.get_dos_on_grid()
                adsorbate_dos = normalise_na_quantities(adsorbate_dos)
                Delta = hybridisation.get_Delta_on_grid()
                Delta = normalise_na_quantities(Delta)

                ax[k,j].plot(EPS_VALUES, adsorbate_dos, color=color_ads[i])
                ax[k,j].plot(EPS_VALUES, Delta,  color='k', alpha=0.5)
                # Annotate the metal name in the top left
                ax[k,j].text(0.05, 0.95, metal, transform=ax[k,j].transAxes, fontsize=12,
                                verticalalignment='top', horizontalalignment='left')

                occupancy = hybridisation.get_occupancy()
                # Annotate the occupancy on the top right corner of the plot
                ax[k,j].annotate(f"{occupancy:.2f}", xy=(0.95, 0.8-0.3*i), xycoords='axes fraction',
                                xytext=(-5, 5), textcoords='offset points',
                                ha='right', va='top', color=color_ads[i], fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
                )

    # Plot the legend for C and O
    ax[0,-1].plot([], [], color=O_COLOR, label='O')
    ax[0,-1].plot([], [], color=C_COLOR, label='C')
    ax[0,-1].legend(loc='upper right', fontsize=12)
    ax[0,-1].axis('off')                
    ax[0,0].set_title('3$d$')
    ax[0,1].set_title('4$d$')
    ax[0,2].set_title('5$d$')

    for a in ax.flatten():
        a.set_xlim([-20, 20])
        a.set_yticks([])
    fig.savefig(f"output/solution_dos_{COMP_SETUP[CHOSEN_SETUP]}.png")