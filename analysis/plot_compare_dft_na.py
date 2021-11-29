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
    return quantity + x_add

if __name__ == "__main__":
    # Use these metals only
    METAL = ['W', 'Re', 'Ag']
    FUNCTIONAL = 'PBE_scf'

    # Get a cycle of with colormap
    colors =  plt.cm.viridis(np.linspace(0, 1, len(METAL)))
    
    # One column for DFT density of states and 
    # one for the Newns-Anderson density of states for C
    # and one for the Newns-Anderson density of states for O
    fig, ax = plt.subplots(1, 3, figsize=(10, 6), constrained_layout=True)

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

    x_pos = 0.0
    x_pos_na = np.zeros(2)
    # Plot the projected density of states
    for i, metal in enumerate(METAL):
        # Normalize the density of states quantities
        energies, pdos_metal_unnorm, _ = pdos_data['slab'][metal]
        energies_C, pdos_C_unnorm = pdos_data['C'][metal]
        energies_O, pdos_O_unnorm = pdos_data['O'][metal]
        x_pos += np.max(pdos_metal_unnorm)
        pdos_metal = normalise_na_quantities(pdos_metal_unnorm, x_pos)
        pdos_C = normalise_na_quantities(pdos_C_unnorm, x_pos)
        pdos_O = normalise_na_quantities(pdos_O_unnorm, x_pos)

        # Plot the projected density of states
        ax[0].plot(energies, pdos_metal, color=colors[i])
        ax[0].fill_between(energies, x_pos, pdos_metal, color=colors[i], alpha=0.25)
        ax[0].plot(energies_C, pdos_C, color='tab:grey', alpha=0.75)
        ax[0].plot(energies_O, pdos_O, color='tab:red', alpha=0.75)
        ax[0].annotate(metal, xy=(-8.5, pdos_metal[-1]+0.5), color=colors[i])

        # Plot the Newns-Anderson projected density of states
        filling = metal_parameters['filling'][metal] 
        eps_d = data_from_dos_calculation[metal]['d_band_centre'] 
        width = data_from_dos_calculation[metal]['width']
        for a, adsorbate in enumerate(['C', 'O']):
            if adsorbate == 'C':
                Vaksq = c_parameters['beta'] * metal_parameters['Vsdsq'][metal]
                Vak = np.sqrt(Vaksq)
                print(f"C adsorbate: Vak: {Vak} for metal: {metal}")
                eps_a = c_parameters['eps_a']
                delta0 = c_parameters['delta0']
                color = 'tab:grey'
            elif adsorbate == 'O':
                Vaksq = o_parameters['beta'] * metal_parameters['Vsdsq'][metal]
                Vak = np.sqrt(Vaksq)
                print(f"O adsorbate: Vak: {Vak} for metal: {metal}")
                eps_a = o_parameters['eps_a']
                delta0 = o_parameters['delta0']
                color='tab:red'
            else:
                raise ValueError('adsorbate must be either C or O')

            hybridisation = NewnsAndersonNumerical(
                Vak = Vak,
                eps_a = eps_a,
                eps_d = eps_d,
                width = width,
                eps = np.linspace(-15, 15, 1000),
                Delta0_mag = delta0,
            )
            hybridisation.calculate_energy()

            Delta = hybridisation.get_Delta_on_grid()
            max_height_Delta = np.max(Delta)
            x_pos_na[a] += 2*max_height_Delta

            Delta = normalise_na_quantities( Delta, x_pos_na[a] )
            # Get the Hilbert transform
            Lambda = hybridisation.get_Lambda_on_grid()
            Lambda = normalise_na_quantities( Lambda, x_pos_na[a] )
            # Get the line representing the eps - eps_a state
            eps_a_line = hybridisation.get_energy_diff_on_grid() 
            eps_a_line = normalise_na_quantities( eps_a_line, x_pos_na[a] )

            ax[a+1].plot(hybridisation.eps, Delta, color=colors[i])
            ax[a+1].fill_between(hybridisation.eps, x_pos_na[a], Delta, color=colors[i], alpha=0.25)
            ax[a+1].plot(hybridisation.eps, Lambda, color=colors[i])

            # Get the adsorbate density of states
            dos = hybridisation.get_dos_on_grid()
            dos *= max_height_Delta / np.max(dos) 
            dos = normalise_na_quantities( dos, x_pos_na[a] )
            # Make the max value of dos the same as Delta
            ax[a+1].plot(hybridisation.eps, dos, color=color)
            # Find the index for eps - eps_a such that
            # the max value is capped at +- max(Delta)
            # index_epsa = np.argwhere(np.abs(eps_a_line) < np.max(Lambda))
            # ax[1].plot(hybridisation.eps, eps_a_line, color=color, alpha=0.5)

    ax[0].set_xlabel('$\epsilon - \epsilon_{F}$ (eV)')
    ax[0].set_ylabel(r'$\rho^{\mathregular{DFT}}$ (states/eV)')
    ax[0].set_xlim(-15, 10)
    ax[1].set_xlim(-15, 10)
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    ax[1].set_ylabel(r'$\rho^{\mathregular{Newns-Anderson}}$ (states/eV)')
    ax[1].set_xlabel('$\epsilon - \epsilon_{F}$ (eV)')
    ax[2].set_xlabel('$\epsilon - \epsilon_{F}$ (eV)')
    ax[0].plot([], [], '-', color='tab:red', label=r'$p-$states O*')
    ax[0].plot([], [], '-', color='tab:grey', label=r'$p-$states C*')
    ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    fig.savefig('output/compare_dft_na.png') 