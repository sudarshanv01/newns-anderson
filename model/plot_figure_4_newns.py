"""Recreate figure 4 of the Newns paper."""

import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
from matplotlib.pyplot import cm
get_plot_params()


if __name__ == '__main__':
    """Create figure 4 of Newns paper.
    Plot the analytical solution for the energy against the adsorbate
    energy for different beta, which is the metal width, for two different
    Fermi levels.
    """
    # Define the parameters
    BETA = [  2,  6, 10, 20, 60, 100] # interaction with metal atoms, eV
    METAL_ENERGIES = 0.0 # center of d-band, eV
    BETA_PRIME = 2 # Interaction with the adsorbate, eV 
    ADSORBATE_ENERGIES = np.linspace(-1, 1, 25) # 2beta' CONVERT UNITS
    EPSILON_RANGE = np.linspace(-10, 10 , 10000) # 2beta CONVERT UNITS
    FERMI_LEVEL = [0.0, 0.9]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5), constrained_layout=True)
    colors = cm.viridis(np.linspace(0, 1, len(BETA)))

    for e_fermi in FERMI_LEVEL:
        for b, beta in enumerate(BETA): 
            all_energies = []

            # prepare the unit coverter from beta to beta_p
            convert = beta / BETA_PRIME
            for d, eps_sigma in enumerate(ADSORBATE_ENERGIES):

                eps_sigma_eV = eps_sigma * 2 * BETA_PRIME + e_fermi
                eps_range = EPSILON_RANGE * 2 * beta

                newns = NewnsAndersonAnalytical(beta = beta, 
                                                beta_p = BETA_PRIME/beta,
                                                eps_d = METAL_ENERGIES,
                                                eps_sigma = eps_sigma_eV,
                                                eps = eps_range,
                                                fermi_energy=e_fermi)

                eps_a_wrt_fermi = newns.eps_sigma * convert - e_fermi / 2 / BETA_PRIME
                all_energies.append( [ eps_a_wrt_fermi, newns.DeltaE * convert ] )

                if newns.has_localised_occupied_state_positive:
                    ax.plot(eps_a_wrt_fermi, newns.DeltaE * convert, '*', color='k')
                if newns.has_localised_occupied_state_negative:
                    ax.plot(eps_a_wrt_fermi, newns.DeltaE * convert, 'v', color='k')

            all_a, all_hyb = np.array(all_energies).T
            all_a_sorted = all_a[np.argsort(all_a)]
            all_hyb_sorted = all_hyb[np.argsort(all_a)]

            if e_fermi == 0: 
                ax.plot(all_a_sorted, all_hyb_sorted, 'o-',
                        color=colors[b],
                        label=r"$\epsilon_f$ = %1.2f, $\beta$ = %1.2f"%(e_fermi, beta))
            else:
                ax.plot(all_a_sorted, all_hyb_sorted, '--',
                        color=colors[b],
                        label=r"$\epsilon_f$ = %1.2f, $\beta$ = %1.2f"%(e_fermi, beta))

            ax.axvline(0, color='k', ls='--')

    ax.set_xlabel(r"$\epsilon_a $ ($2 \beta'$) ")
    ax.set_ylabel(r" $\Delta E$ ($2 \beta'$)")
    ax.set_xlim([-1.2, 1.2])
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    fig.savefig('output/figure_4_newns.png')