"""Recreate Figure 6 of the Newns paper."""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import yaml 
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':
    """Recreate Figure 6 of the Newns paper. 
    The plot has the adsorption energy for H* for four
    different metals Ni, Cr, Ti and Cu."""

    # Load data
    parameters = yaml.safe_load(open('parameters_figure_6.yaml'))

    # Over a range of beta_primes
    beta_prime = np.linspace(1, 7, 10)

    # The adsorbate state is chosen as 
    eps_a = -13.6 # eV

    # Reference everything to the Fermi level
    eps_d = 0.0

    # Coulomb term
    U = 12.9 # eV

    # Range of energies
    EPSILON_RANGE = np.linspace(-25, 15, 1000) # range of energies plot in dos

    # Figure to plot the energies
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    # Plot the expectation value of the spin up and spin down components
    figs, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = cm.viridis(np.linspace(0, 1, len(parameters) ))

    for i, metal in enumerate(parameters):

        print(f'Working on {metal}')
        all_energies = []
        all_spin_up = []
        all_spin_down = []

        # For each metal also plot the energy on a grid
        figg, axg = plt.subplots(1, len(beta_prime), 
                            figsize=(5*len(beta_prime), 4.5), constrained_layout=True)

        for j, beta_p in enumerate(beta_prime):
            beta = parameters[metal]['width']
            fermi_energy = parameters[metal]['fermi_level']
            eps_a_wrt_fermi = eps_a - fermi_energy / 2 / beta
            newns = NewnsAndersonAnalytical(beta = beta, 
                                            beta_p = beta_p / 2 / beta, 
                                            eps_d = eps_d,
                                            eps_a = eps_a,
                                            eps = EPSILON_RANGE,
                                            fermi_energy = fermi_energy,
                                            U = U)
            newns.self_consistent_calculation()
            # The quantity that we want to plot
            energy_in_eV = newns.DeltaE * 2 * beta
            all_energies.append(energy_in_eV)
            all_spin_up.append(newns.n_plus_sigma)
            all_spin_down.append(newns.n_minus_sigma)

            axg[j].set_title(f"$\\beta' = {beta_p:1.2f}$")
            cax = axg[j].imshow(newns.energies_grid, extent=[0, 1, 0, 1], origin='lower',)
            axg[j].set_xlabel(r'$n_+$')
            axg[j].set_ylabel(r'$n_{-}$')
            # plot colorbar
            figg.colorbar(cax, ax=axg[j])

        # Plot the energies against beta_prime
        ax.plot(beta_prime, all_energies, 'o-', label=metal, color=colors[i])
        axs[0].plot(beta_prime, all_spin_up, 'o-', label=metal, color=colors[i])
        axs[1].plot(beta_prime, all_spin_down, 'o-', label=metal, color=colors[i])

        # Save the figures for the metals
        figg.savefig(f'output/figure_6/{metal}_energies.png')
    
    ax.legend(loc='best')
    ax.set_xlabel(r"$\beta'$")
    ax.set_ylabel(r'$\Delta E$ (eV)')
    ax.set_title(r'$\Delta E$ for different metals')
    fig.savefig('output/figure_6_newns.png')

    axs[0].legend(loc='best')
    axs[0].set_xlabel(r"$\beta'$ (eV)")
    axs[0].set_ylabel(r'$n_+$')
    axs[1].set_ylabel(r'$n_{-}$')
    figs.savefig('output/figure_6_newns_spin.png')
