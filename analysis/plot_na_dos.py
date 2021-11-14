"""Plot the Newns-Anderson dos with the parameters from the fitting procedure."""
import numpy as np
import json
from NewnsAnderson import NewnsAndersonNumerical, JensNewnsAnderson
from collections import defaultdict
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from adjustText import adjust_text
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

def create_plot_layout(adsorbate):
    """Create a plot layout for plotting the Newns-Anderson
    dos and the energies of orthogonalisation, spd hybridisation
    energy for each specific adsorbate."""
    fig = plt.figure(figsize=(14,12), constrained_layout=True)
    gs = fig.add_gridspec(nrows=12, ncols=3,)
    # The first 2 rows will be the orthogonalisation energies, spd
    # hybridisation energy and the total energy as a function of the
    # d-band centre.
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax2 = fig.add_subplot(gs[0:3, 1])
    ax3 = fig.add_subplot(gs[0:3, 2])
    # Set the axes labels
    ax1.set_xlabel('d-band centre (eV)')
    ax1.set_ylabel('NA energy (eV)')
    ax2.set_xlabel('d-band centre (eV)')
    ax2.set_ylabel('Ortho energy (eV)')
    ax3.set_xlabel('d-band centre (eV)')
    ax3.set_ylabel('Total energy (eV)')
    # Then make three plots with the density of states coming from
    # the different solutions of the Newns-Anderson equation.
    ax4 = fig.add_subplot(gs[3:, 0])
    ax5 = fig.add_subplot(gs[3:, 1])
    ax6 = fig.add_subplot(gs[3:, 2])
    # Set the axes labels
    ax4.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax5.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax6.set_xlabel(r'$\epsilon - \epsilon_{F}$ (eV)')
    ax4.set_ylabel('Projected Density of States')
    # Remove y-ticks from 4, 5, 6
    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])
    return fig, np.array([ [ax1, ax2, ax3], [ax4, ax5, ax6] ])

def normalise_na_quantities(quantity, x_add):
    """Utility function to align the density of states for Newns-Anderson plots."""
    return quantity / np.pi**2 + x_add

if __name__ == '__main__':
    """Get the Newns-Anderson dos and occupancy from the fitting procedure."""

    FUNCTIONAL = 'PBE_scf'
    REMOVE_LIST = [ 'Y', 'Sc', 'Nb', 'Hf', 'Ti', 'Os', 'Co' ] 
    KEEP_LIST = []

    # Read in scaling parameters from the model.
    with open(f"output/O_parameters_{FUNCTIONAL}.json", 'r') as f:
        o_parameters = json.load(f)
    with open(f"output/C_parameters_{FUNCTIONAL}.json", 'r') as f:
        c_parameters = json.load(f)
    adsorbate_parameters = {'O': o_parameters, 'C': c_parameters}
    # Input parameters to help with the dos from Newns-Anderson
    data_from_dos_calculation = json.load(open(f'output/pdos_moments_{FUNCTIONAL}.json')) 
    data_from_energy_calculation = json.load(open(f'output/adsorption_energies_{FUNCTIONAL}.json'))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))

    # Create a separate graph for each parameter.
    for adsorbate in adsorbate_parameters:
        fig, ax = create_plot_layout(adsorbate)

        # Store texts for annotations
        texts = []

        # Adsorbate specific parameters
        eps_a = adsorbate_parameters[adsorbate]['eps_a']
        alpha = adsorbate_parameters[adsorbate]['alpha']
        beta = adsorbate_parameters[adsorbate]['beta']
        delta0 = adsorbate_parameters[adsorbate]['delta0']

        x_pos = 0.0

        for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            # Iterate over all the metals
            for i, metal in enumerate(metal_row):
                # Metal specific parameters
                try:
                    Vak = np.sqrt(beta) * np.sqrt(data_from_LMTO['Vsdsq'][metal])
                    eps_d = data_from_dos_calculation[metal]['d_band_centre']
                    width = data_from_dos_calculation[metal]['width']
                    # dE = data_from_energy_calculation[adsorbate][metal]
                except KeyError:
                    continue

                hybridisation = NewnsAndersonNumerical(
                    Vak = Vak,
                    eps_a = eps_a,
                    eps_d = eps_d,
                    width = width,
                    eps = np.linspace(-15, 15, 1000),
                    k = delta0,
                )
                hybridisation.calculate_energy()
            
                # Decide on the x-position based on the d-band centre
                x_pos += 2 * np.max(hybridisation.Delta) / np.pi**2

                # Get the metal projected density of states
                Delta = normalise_na_quantities( hybridisation.Delta, x_pos )
                # Get the hilbert transform
                Lambda = normalise_na_quantities( hybridisation.Lambda, x_pos )
                # Get the line representing the eps - eps_a state
                eps_a_line = hybridisation.eps - hybridisation.eps_a
                eps_a_line = normalise_na_quantities( eps_a_line, x_pos )

                # Get the adsorbate density of states
                na = hybridisation.dos + x_pos 

                # ax[1,j].plot(hybridisation.eps, Delta, color='tab:red', lw=3)
                ax[1,j].plot(hybridisation.eps, na, color='tab:blue')
                # ax[1,j].plot(hybridisation.eps, Lambda, color='tab:orange', lw=3, alpha=0.25)
                # ax[1,j].plot(hybridisation.eps, eps_a_line, color='tab:green', lw=3, alpha=0.25)

                # Get the different components of the energy by creating
                # an instance of the JensNewnsAnderson class.
                jna = JensNewnsAnderson(
                    Vsd = [ np.sqrt( data_from_LMTO['Vsdsq'][metal] ) ],
                    eps_a = eps_a,
                    width = [ width ],
                    filling = [ data_from_LMTO['filling'][metal] ],
                )
                total_energy = jna.fit_parameters(eps_ds = [ eps_d ], alpha=alpha, beta=beta, constant=delta0)
                spd_hyb_energy = jna.spd_hybridisation_energy
                ortho_energy = jna.ortho_energy

                # Plot the energies in the different graphs against the d-band centre                
                if metal in FIRST_ROW:
                    colour = 'red'
                elif metal in SECOND_ROW:
                    colour = 'orange'
                elif metal in THIRD_ROW:
                    colour = 'green'

                ax[0,0].plot(eps_d, spd_hyb_energy, 'o', color=colour)
                ax[0,1].plot(eps_d, ortho_energy, 'o', color=colour)
                ax[0,2].plot(eps_d, total_energy, 'o', color=colour)

                texts.append([
                    ax[0,0].text(eps_d, spd_hyb_energy, metal, color=colour, fontsize=12),
                    ax[0,1].text(eps_d, ortho_energy, metal, color=colour, fontsize=12),
                    ax[0,2].text(eps_d, total_energy, metal, color=colour, fontsize=12),
                ])

        # Add the text to the plot
        texts = np.array(texts).T
        for i, text in enumerate(texts):
            adjust_text(text, ax=ax[0,i]) 

        fig.savefig(f'output/optimised_parameters_dos_{adsorbate}.png', dpi=300)
