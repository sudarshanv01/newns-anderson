"""Recreate Figure 1 of the Newns paper."""

import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':
    """Recreate Figure 1 of the Newns paper.
    The units along both the dos and energy axis are -2beta, 
    where beta is the interaction of the chain with itself. More
    specifically it is the width of the d-band.
    """
    # Parameters to recreate the figure
    EPSILON_RANGE = np.linspace(-2.5, 2.5, 1000) # in units of -2beta 
    BETA_PRIME = [0.5, 1.5] # in units of -2beta 
    # Choose epsilon_{sigma} such that it resemble the plots in the figues
    EPSILON_A = [
        [ -1.5, 0.75 ],
        [ -0.25             ],
    ] 
    EPSILON_D = 0 
    U = 0 # There is no coulomb interaction
    FERMI_ENERGY = 0 # No fermi energy shift

    # Create the figure
    fige, axe = plt.subplots(2, 1, figsize=(6, 10), constrained_layout=True)
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

    for i, beta_p in enumerate(BETA_PRIME):
        print('beta prime = {} (units: 2beta)'.format(beta_p))
        # Choose only the right epsilon_{sigma}
        for j, _epsilon_a in enumerate(EPSILON_A[i]):

            # All energy quantities in eV
            epsilon_a = _epsilon_a * 2 * beta_p 
            epsilon_d = EPSILON_D 
            epsilon_range = EPSILON_RANGE * 2 * beta_p

            analytical = NewnsAndersonAnalytical(beta_p=beta_p,
                                                beta=beta_p, 
                                                eps_a=epsilon_a, 
                                                eps_d=epsilon_d, 
                                                eps=epsilon_range,
                                                U=U,
                                                fermi_energy=FERMI_ENERGY)

            analytical.self_consistent_calculation()
            # Plot in terms of 2beta
            if j == 0:
                axe[i].plot(analytical.eps, analytical.Delta, '--', lw=3, color='k', label=r"$\Delta$")
                axe[i].plot(analytical.eps, analytical.Lambda, '-', lw=3, color='k', label=r"$\Lambda$")
                axe[i].set_title(r"$\beta'=%1.2f$"%(beta_p))
            axe[i].plot(analytical.eps, analytical.eps - analytical.eps_sigma, '-', lw=3, color=colors[j])
            axe[i].plot(analytical.eps, analytical.rho_aa, '-', lw=3, color=colors[j])
                    # label=r"$\epsilon_{\sigma}=%1.2f$"%(analytical.eps_sigma), alpha=0.5)
            axe[i].set_ylim([-3*beta_p**2, 3*beta_p**2])

            if analytical.has_localised_occupied_state_positive:
                axe[i].plot(analytical.root_positive, 0, '*', color=colors[j], ms=16)
                axe[i].axvline(analytical.root_positive, color=colors[j], ls='-.', alpha=0.25)
                print(f'Occupancy of the occupied state (positive root) is {analytical.na_sigma_pos}')
            if analytical.has_localised_occupied_state_negative:
                axe[i].plot(analytical.root_negative, 0, '*', color=colors[j], ms=16)
                axe[i].axvline(analytical.root_negative, color=colors[j], ls='-.', alpha=0.25)
                print(f'Occupancy of the occupied state (negative root) is {analytical.na_sigma_neg}')
            
        if i == 1:
            axe[i].legend(loc='best')

        for a in axe:
            a.set_ylabel(r'$\rho_{aa}^{\sigma}$ ($2\beta$)')
            a.set_xlabel(r'$\epsilon (2\beta) $')
            # a.legend(loc='best')
            a.set_xlim([-2.5, 2.5])

    fige.savefig('output/figure_1_newns.png')

        



