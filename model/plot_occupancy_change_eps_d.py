"""Plot the variation of the occupancy of a single state against the d-band center."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()


if __name__ == '__main__':
    """Plot the variation of the occupancy (degenerate) of a single state as a 
    function of the change in the d-band centre. This script is an analogue of the
    variation with the energy."""
    EPSILON_RANGE = np.linspace(-15, 15, 4000) # range of energies plot in dos
    BETA_PRIME = [1, 2] # Interaction of metal and adsorbate in 2beta units 
    EPSILON_SIGMA = [ -4, 2.5 ] # renormalised energy of adsorbate
    EPSILON_D = np.linspace(-8, 2) # Band center in eV 
    BETA = 1 # in units of eV
    NUM_DENSITY_OF_STATES = 5 # Number of density of states to plot
    colors = cm.RdBu(np.linspace(0, 1, len(EPSILON_SIGMA) * len(BETA_PRIME)))
    FERMI_ENERGY = 0.0 # Fermi energy in 2beta units
    U = 0.0 # no coulomb interaction

    # Plot the energies in this figure
    fige, axe = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)

    # Plot the components of the energy in this figure
    figs, axs = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # eps_sigma here would be for different adsorbates
    index = 0
    for s, eps_sigma in enumerate(EPSILON_SIGMA):
        for b, beta_p in enumerate(BETA_PRIME):

            all_occupancy = []

            for d, eps_d in enumerate(EPSILON_D):
                newns = NewnsAndersonAnalytical(beta = BETA, 
                                                beta_p = beta_p / 2 / BETA, 
                                                eps_d = eps_d,
                                                eps_a = eps_sigma,
                                                eps = EPSILON_RANGE,
                                                fermi_energy = FERMI_ENERGY,
                                                U = U)
                newns.self_consistent_calculation()

                occupancy = 0.0
                if newns.has_localised_occupied_state_positive:
                    occupancy += newns.na_sigma_pos
                if newns.has_localised_occupied_state_negative:
                    occupancy += newns.na_sigma_neg

                energy_in_eV = newns.eps_d * 2 * BETA

                # The quantity that we want to plot
                all_occupancy.append( [ energy_in_eV, occupancy ] )

            # Plot the energies 
            eps_a_in_eV = newns.eps_sigma * 2 * BETA
            all_occupancy = np.array(all_occupancy).T
            axe.plot( all_occupancy[0], all_occupancy[1], '-o', lw=3, color=colors[index],
                     label = r"$V_{\rm ak} = %1.2f$ eV, $\epsilon_a = %1.2f$ eV"%(beta_p, eps_a_in_eV))
            index += 1


    axe.set_xlabel(r'$\epsilon_d$ (eV) ')
    axe.set_ylabel(r'$\left < n_a \right>$ (e)')
    # axe.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    axe.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
    fige.savefig('output/NewnsAnderson_vary_occupancy_eps_d.png')




