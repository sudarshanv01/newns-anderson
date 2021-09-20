"""Recreate Figure 6 of the Newns paper."""

import numpy as np
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
    beta_prime = np.linspace(1, 7, 100)

    # The adsorbate state is chosen as 
    eps_sigma = 0.0 #-14.3 # eV
    # Reference everything to the Fermi level
    eps_d = 0.0
    # Range of energies
    EPSILON_RANGE = np.linspace(-25, 15, 1000) # range of energies plot in dos

    # Figure to plot the energies
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    for metal in parameters:
        print(f'Working on {metal}')
        all_energies = []
        for beta_p in beta_prime:
            beta = parameters[metal]['width']
            fermi_energy = parameters[metal]['fermi_level']
            newns = NewnsAndersonAnalytical(beta = beta, 
                                            beta_p = beta_p/2/beta, 
                                            eps_d = eps_d,
                                            eps_sigma = eps_sigma,
                                            eps = EPSILON_RANGE,
                                            fermi_energy = fermi_energy/2/beta)
            # The quantity that we want to plot
            energy_in_eV = newns.DeltaE * 2 * beta
            all_energies.append(energy_in_eV)
        # Plot the energies against beta_prime
        ax.plot(beta_prime, all_energies, '-', label=metal)
    
    ax.legend(loc='best')
    ax.set_xlabel(r"$\beta'$")
    ax.set_ylabel(r'$\Delta E$ (eV)')
    ax.set_title(r'$\Delta E$ for different metals')
    fig.savefig('output/figure_6_newns.png')

