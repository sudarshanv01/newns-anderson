"""Compare the density of states coming from DFT and NA."""
import matplotlib.pyplot as plt
import numpy as np
import json
from norskov_newns_anderson.NewnsAnderson import NorskovNewnsAnderson, NewnsAndersonNumerical
from plot_params import get_plot_params
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

def normalise_na_quantities(quantity, x_add):
    """Utility function to align the density of states for Newns-Anderson plots."""
    return quantity / np.max(quantity) + x_add

if __name__ == "__main__":
    # Use these metals only
    METALS = [FIRST_ROW, SECOND_ROW, THIRD_ROW] 
    FUNCTIONAL = 'PBE_scf'

    # Get a cycle of with colormap
    colors =  plt.cm.viridis(np.linspace(0, 1, len(FIRST_ROW)))

    # Plot the sp projected density of states from DFT
    # and also the projected density of states from the Newns-Anderson model 
    # For each row separately
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True)

    # Load the projected density of states from 
    # a DFT calculation
    pdos_data = json.load(open(f'output/pdos_{FUNCTIONAL}.json'))
    # load the energies and the C, O projected density of states

    # Load the data to make the projected density of states
    # from the Newns-Anderson model
    with open(f"output/O_parameters_{FUNCTIONAL}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{FUNCTIONAL}.json", 'r') as f:
        c_parameters = json.load(f)
    with open(f"inputs/data_from_LMTO.json", 'r') as f:
        metal_parameters = json.load(f)
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{FUNCTIONAL}.json')) 

    # Plot the projected density of states
    for row_index, row_elements in enumerate(METALS):
        for i, metal in enumerate(row_elements):

            # Normalize the density of states quantities
            try:
                energies, pdos_metal_unnorm, _ = pdos_data['slab'][metal]
            except KeyError:
                continue
            energies_C, pdos_C_unnorm = pdos_data['C'][metal]
            energies_O, pdos_O_unnorm = pdos_data['O'][metal]

            # Shift the graph up by this much
            x_pos = 2 * i
            pdos_metal = normalise_na_quantities(pdos_metal_unnorm, x_pos)
            pdos_C = normalise_na_quantities(pdos_C_unnorm, x_pos)
            pdos_O = normalise_na_quantities(pdos_O_unnorm, x_pos)

            # Plot the projected density of states
            # ax[row_index].plot(energies, pdos_metal, color=colors[i])
            # ax[row_index].fill_between(energies, x_pos, pdos_metal, color=colors[i], alpha=0.25)
            ax[row_index].plot(energies_C, pdos_C, color='tab:grey', alpha=0.75)
            ax[row_index].plot(energies_O, pdos_O, color='tab:red', alpha=0.75)
            ax[row_index].annotate(metal, xy=(-8.5, pdos_metal[-1]+0.5), color=colors[i])

            # Plot the Newns-Anderson projected density of states
            filling = metal_parameters['filling'][metal] 
            eps_d = data_from_dos_calculation[metal]['d_band_centre'] 
            width = data_from_dos_calculation[metal]['width']
            for a, adsorbate in enumerate(['O', 'C']):
                if adsorbate == 'C':
                    Vak = c_parameters['Vak'][metal]
                    print(f"C adsorbate: Vak: {Vak} for metal: {metal}")
                    eps_a = c_parameters['eps_a']
                    delta0 = c_parameters['delta0']
                    alpha = c_parameters['alpha']
                    color = 'tab:grey'
                elif adsorbate == 'O':
                    Vak = c_parameters['Vak'][metal]
                    print(f"O adsorbate: Vak: {Vak} for metal: {metal}")
                    eps_a = c_parameters['eps_a']
                    delta0 = c_parameters['delta0']
                    alpha = c_parameters['alpha']
                    color = 'tab:red'
                else:
                    raise ValueError('adsorbate must be either C or O')

                hybridisation = NewnsAndersonNumerical(
                    Vak = Vak,
                    eps_a = eps_a,
                    eps_d = eps_d,
                    width = width,
                    eps = np.linspace(-20, 20, 1000),
                    Delta0_mag = delta0,
                )
                hybridisation.calculate_energy()

                # Get the adsorbate density of states
                dos = hybridisation.get_dos_on_grid()
                # dos *= max_height_Delta / np.max(dos) 
                dos = normalise_na_quantities( dos, x_pos )
                # Make the max value of dos the same as Delta
                ax[row_index].plot(hybridisation.eps, dos, color=color)

    for a in ax:
        a.set_xlim(-10, 10)
        a.set_xlabel('$\epsilon - \epsilon_F$ (eV)')
        a.set_yticks([])
    ax[0].set_ylabel('Projected density of states')
    fig.savefig('output/compare_dft_na.png') 