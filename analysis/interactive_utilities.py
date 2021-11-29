"""Functions for interactive plots of the Newns-Anderson model."""
from collections import defaultdict
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import inspect
import functools
from adjustText import adjust_text
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical, NorskovNewnsAnderson

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

def interactive_newns_anderson_dos(Vak=1, eps_a=-1, eps_d=-5, width=4, 
                                   Delta0_mag=2, eps_sp_min=-15, eps_sp_max=15):
    """Function to interactively plot the Newns-Anderson model density of states."""
    eps = np.linspace(-40, 40, 1000)

    hybridisation = NewnsAndersonNumerical(
        Vak = Vak,
        eps_a = eps_a,
        eps = eps,
        width = width,
        eps_d = eps_d,
        Delta0_mag = Delta0_mag,
        eps_sp_max = eps_sp_max,
        eps_sp_min = eps_sp_min,
    )
    # Run the calculation
    hybridisation.calculate_energy()
    hybridisation.calculate_occupancy()

    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Get the Newns-Anderson model outputs
    Delta = hybridisation.get_Delta_on_grid()
    Lambda = hybridisation.get_Lambda_on_grid()
    eps_diff = hybridisation.get_energy_diff_on_grid()
    pdos = hybridisation.get_dos_on_grid()

    # Plot the Newns-Anderson model
    ax[0].plot(eps, Delta, color='tab:red', lw=3) 
    ax[0].plot(eps, Lambda, color='tab:green', lw=3)
    # ax[0].plot(eps, eps_diff, color='tab:red', lw=3, alpha=0.25)
    ax[1].plot(eps, pdos, color='tab:blue', lw=3)

    ax[0].set_xlabel(r'$\epsilon - \epsilon_F$ (eV)')
    ax[1].set_xlabel(r'$\epsilon - \epsilon_F$ (eV)')
    ax[0].set_ylabel(r'$\Delta, \Lambda$')
    ax[1].set_ylabel(r'Adsorbate pdos')

    plt.show()

def interactive_newns_anderson_energy(Vak=1, eps_a=-1, width=4, 
                                   Delta0_mag=2, eps_sp_min=-15, eps_sp_max=15):
    """Plot the Newns Anderson energy variation and the occupancy 
    against the d-band center and the corresponding density of states
    along the different eps_ds chosen."""
    # Plot the energies, occupancies and the density of states
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    # Standard range of the d-band centres
    EPS_D_RANGE = np.linspace(-4, 0.5, 10)
    EPS_RANGE = np.linspace(-20, 20, 1000)
    
    # Get the energy and occupancy for each epsilon value
    energy = np.zeros(len(EPS_D_RANGE))
    occupancy = np.zeros(len(EPS_D_RANGE))

    # Store the position to plot the density of states
    x_pos = 0.0

    for i, eps_d in enumerate(EPS_D_RANGE): 
        hybridisation = NewnsAndersonNumerical(
            Vak = Vak,
            eps_a = eps_a,
            eps = EPS_RANGE,
            width = width,
            eps_d = eps_d,
            Delta0_mag = Delta0_mag,
            eps_sp_max = eps_sp_max,
            eps_sp_min = eps_sp_min,
        )
        # Run the calculation
        hybridisation.calculate_energy()
        hybridisation.calculate_occupancy()
        energy[i] = hybridisation.get_energy()
        occupancy[i] = hybridisation.get_occupancy()

        # Get the density of states
        Delta = hybridisation.get_Delta_on_grid()
        max_height_Delta = np.max(Delta)
        x_pos += 2 * max_height_Delta
        Lambda = hybridisation.get_Lambda_on_grid()
        pdos = hybridisation.get_dos_on_grid()
        pdos *= max_height_Delta / np.max(pdos) 

        # Normalise the quantities
        Delta = normalise_na_quantities(Delta, x_pos)
        Lambda = normalise_na_quantities(Lambda, x_pos)
        pdos = normalise_na_quantities(pdos, x_pos)

        ax[2].plot(EPS_RANGE, Delta, '-', lw=3, color='tab:red', alpha=0.75)
        ax[2].plot(EPS_RANGE, Lambda, '-', lw=3, color='tab:blue', alpha=0.25)
        ax[2].plot(EPS_RANGE, pdos, '-', lw=3, color='tab:green')
    
    # Plot the energy and occupancy
    ax[0].plot(EPS_D_RANGE, energy, 'o-', color='tab:red', lw=3)
    ax[1].plot(EPS_D_RANGE, occupancy, 'o-', color='tab:blue', lw=3)

    # Set axes labels
    ax[0].set_xlabel(r'$\epsilon_d$ (eV)')
    ax[1].set_xlabel(r'$\epsilon_d$ (eV)')
    ax[0].set_ylabel(r'$\Delta E$ (eV)')
    ax[1].set_ylabel(r'$n_a$ (e)')
    ax[2].set_xlabel(r'$\epsilon - \epsilon_F$ (eV)')
    ax[2].set_ylabel(r'Project Density of States')
    ax[2].set_yticks([])

    plt.show()


def normalise_na_quantities(quantity, x_add):
    """Utility function to align the density of states for Newns-Anderson plots."""
    return quantity + x_add

def interactive_norskov_newns_anderson(adsorbate = 'O', alpha=0.2, beta=0.8,  Delta0_mag=2.0, constant=-1): 
    """Function to interactively visualise the influence of different fitting parameters
    on the energies of the Norksov-Newns-Anderson model."""
    # For each of the three rows of transition metals plot the components 
    # of the energy, the sum of occupancy and filling and the orthogonalisation
    # element and the projected density of states.
    fig, ax = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True)
    ax[0,0].set_ylabel(r'Energy (eV)')
    ax[1,0].set_ylabel(r'$n_a + f$', color='k')
    ax[2,0].set_ylabel(r'DOS (eV$^{-1}$)')
    # Also plot the parity plot between the hybridisation energy from
    # the model and that from DFT calculations.
    figp, axp = plt.subplots(1, 1, figsize=(8,6), constrained_layout=True)
    axp.set_ylabel(r'Hybridisation Energy (eV)')
    axp.set_xlabel(r'DFT Energy (eV)')
    # We will consider all metals in the plot and remove all metals in
    # the REMOVE_LIST
    METAL_ROWS = [FIRST_ROW, SECOND_ROW, THIRD_ROW]
    REMOVE_LIST = yaml.safe_load(stream=open('remove_list.yaml', 'r'))['remove']
    # Present on the choice of functional
    FUNCTIONAL = 'PBE_scf'
    # Data from LMTO or DFT calculations
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{FUNCTIONAL}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{FUNCTIONAL}.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    with open(f"output/{adsorbate}_parameters_{FUNCTIONAL}.json") as handle:
        adsorbate_parameters = json.load(handle)
        eps_a = adsorbate_parameters['eps_a']
    # Preset values for some fitting parameters
    EPS_VALUES = np.linspace(-20, 20, 1000)
    EPS_SP_MAX = 15
    EPS_SP_MIN = -15
    CONSTANT_OFFSET = constant
    color_row = ['tab:red', 'tab:green', 'tab:blue']

    texts = []
    for row_index, ROW in enumerate(METAL_ROWS):
        parameters = defaultdict(list)
        for metal in ROW: 
            if metal not in data_from_dos_calculation or metal in REMOVE_LIST:
                continue
            width = data_from_dos_calculation[metal]['width']
            parameters['width'].append(width)
            d_band_centre = data_from_dos_calculation[metal]['d_band_centre']
            parameters['d_band_centre'].append(d_band_centre)
            Vsd = np.sqrt(data_from_LMTO['Vsdsq'][metal])
            parameters['Vsd'].append(Vsd)
            parameters['filling'].append(data_from_LMTO['filling'][metal])
            Vak = np.sqrt(beta) * Vsd
            parameters['Vak'].append(Vak)
            parameters['metal'].append(metal)
            parameters['DFT_energies'].append(data_from_energy_calculation[adsorbate][metal])

        # Fit the parameters
        fitting_function =  NorskovNewnsAnderson(
            Vsd = parameters['Vsd'], 
            width = parameters['width'], 
            eps_a = eps_a,
            eps_sp_min = EPS_SP_MIN,
            eps_sp_max = EPS_SP_MAX,
            eps = EPS_VALUES,
            Delta0_mag = Delta0_mag,
            # filling = parameters['filling'],
        )

        # Get the energies
        fitting_function.fit_parameters((alpha, beta, CONSTANT_OFFSET), parameters['d_band_centre'])
        hyb_energies = fitting_function.hybridisation_energy
        ortho_energies = fitting_function.ortho_energy
        spd_hyb_energies = fitting_function.spd_hybridisation_energy

        # Store the occupancy and filling
        filling = fitting_function.filling
        occupancy = fitting_function.na

        # Plot the energies
        ax[0,row_index].plot(parameters['d_band_centre'], 
                             hyb_energies, 
                             marker='o', ls='-', color='k',
                             label='Total')
        ax[0,row_index].plot(parameters['d_band_centre'],
                             spd_hyb_energies, 
                             marker='o', ls='--', color='k',
                             alpha=0.7,
                             label='spd')
        ax2 = ax[0,row_index].twinx()
        ax2.plot(parameters['d_band_centre'], ortho_energies, 'o-', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax[0,row_index].set_xlabel(r'$\epsilon_d$ (eV)')

        # Plot the occupancies and filling
        ax[1,row_index].plot(parameters['d_band_centre'], 
                             occupancy+filling, 
                             marker='o', ls='-', color='k')
        ax[1,row_index].plot(parameters['d_band_centre'], 
                             filling, 
                             marker='o', ls='--', color='k')
        ax[1,row_index].set_xlabel(r'$\epsilon_d$ (eV)')
        ax3 = ax[1,row_index].twinx()
        ax3.plot(parameters['d_band_centre'], alpha*beta*np.array(parameters['Vsd'])**2, 'o-', color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
        if row_index == 2:
            ax3.set_ylabel(r'$\alpha \beta V_{sd}^2$ (eV)', color='tab:green')
            ax2.set_ylabel('Orthogonalisation Energy (eV)', color='tab:green')

        # Get the density of states
        x_pos = 0.0
        for i, metal in enumerate(parameters['metal']): 
            if metal not in data_from_dos_calculation or metal in REMOVE_LIST:
                continue
            hybridisation = NewnsAndersonNumerical(
                Vak = parameters['Vak'][i],
                eps_a = eps_a,
                eps = EPS_VALUES,
                width = width,
                eps_d = parameters['d_band_centre'][i],
                Delta0_mag = Delta0_mag,
                eps_sp_max = EPS_SP_MAX,
                eps_sp_min = EPS_SP_MIN,
            )
            # Run the calculation
            hybridisation.calculate_energy()
            hybridisation.calculate_occupancy()

            Delta = hybridisation.get_Delta_on_grid()
            max_height_Delta = np.max(Delta)
            x_pos += 2 * max_height_Delta
            Lambda = hybridisation.get_Lambda_on_grid()
            pdos = hybridisation.get_dos_on_grid()
            energy_diff = hybridisation.get_energy_diff_on_grid()
            pdos *= max_height_Delta / np.max(pdos) 
            # Annotate the metal 
            ax[2,row_index].annotate(metal,
                                     xy=( 10, x_pos),
                                     fontsize=12,)
            # If there is a pole, mark it with a star
            if hybridisation.poles:
                ax[2,row_index].plot(hybridisation.poles,
                                     x_pos*np.ones(len(hybridisation.poles)),
                                     marker='*', color='k')


            # Normalise the quantities
            Delta = normalise_na_quantities(Delta, x_pos)
            Lambda = normalise_na_quantities(Lambda, x_pos)
            pdos = normalise_na_quantities(pdos, x_pos)
            energy_diff = normalise_na_quantities(energy_diff, x_pos)
            # Plot the energy difference only between -10,10
            index_diff = np.argwhere(np.abs(EPS_VALUES) < 10)


            ax[2,row_index].plot(EPS_VALUES, Delta, '-', color='tab:red', alpha=0.50)
            ax[2,row_index].plot(EPS_VALUES, Lambda, '-', lw=3, color='tab:blue', alpha=0.25)
            ax[2,row_index].plot(EPS_VALUES[index_diff], energy_diff[index_diff], '-', color='tab:green', alpha=0.4)
            ax[2,row_index].plot(EPS_VALUES, pdos, '-', color='k')
            ax[2,row_index].set_yticks([])
            ax[2,row_index].set_xlabel(r'$\epsilon - \epsilon_F$ (eV)')
    
            # Plot the hybridisation energies against the DFT energies
            axp.plot(parameters['DFT_energies'][i], 
                     hyb_energies[i], marker='o', 
                     color=color_row[row_index])
            texts.append(axp.text(parameters['DFT_energies'][i], 
                                    hyb_energies[i], 
                                    metal, color=color_row[row_index]))

    adjust_text(texts, ax=axp) 
    # Plot the parity line for axp based on the x-axis
    axp.plot(axp.get_xlim(), axp.get_xlim(), 'k--')
    plt.show()