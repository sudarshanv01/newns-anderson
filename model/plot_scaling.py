"""Plot the variation of energies of one intermediate against another, scaling."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
from collections import defaultdict
get_plot_params()

if __name__ == '__main__':
    """Plot the variation of the two energies against each other based on
    different values of the d-band center."""
    EPSILON_RANGE = np.linspace(-15, 15, 4000) # range of energies plot in dos
    BETA_PRIME = [ 2, 1.6, 2.5 ] # Lower average and upper bound of the beta
    assert len(BETA_PRIME) == 3, "BETA_PRIME must be a list of length 3"

    # Scale these two energies
    EPSILON_SIGMA_all = [ [ -2, -3 ], [ 1, 2.0 ], [-3, 2] ] # renormalised energy of adsorbate
    EPSILON_D = np.linspace(-8, -2) # Band center in eV 
    BETA = 2 # in units of eV
    FERMI_ENERGY = 0.0 # Fermi energy in 2beta units
    U = 0.0 # no coulomb interaction

    # Plot the energies in this figure
    fige, axe = plt.subplots(1, len(EPSILON_SIGMA_all), figsize=(15, 6), constrained_layout=True)

    axe[0].set_title('Both $\epsilon_a < 0$')
    axe[1].set_title('Both $\epsilon_a > 0$')
    axe[2].set_title('Mixed')

    # Plot the energies for different eps_SIGMA
    for j, EPSILON_SIGMA in enumerate(EPSILON_SIGMA_all):
        ax = axe[j]
        colors = cm.RdBu(np.linspace(0, 1, len(EPSILON_SIGMA) * len(BETA_PRIME)))
        # eps_sigma here would be for different adsorbates
        index = 0
        # Store all the scaling data in this dict
        data_scaling = defaultdict(list)
        for s, eps_sigma in enumerate(EPSILON_SIGMA):
            for b, beta_p in enumerate(BETA_PRIME):

                all_energies = []
                all_eps_sigma_pos = []
                all_eps_sigma_neg = []
                all_tan_comp = []

                energies_dband = []
                energies_chemisorp = []

                for d, eps_d in enumerate(EPSILON_D):
                    newns = NewnsAndersonAnalytical(beta = BETA, 
                                                    beta_p = beta_p / 2 / BETA, 
                                                    eps_d = eps_d,
                                                    eps_a = eps_sigma,
                                                    eps = EPSILON_RANGE,
                                                    fermi_energy = FERMI_ENERGY,
                                                    U = U)
                    newns.self_consistent_calculation()
                    energy_in_eV = newns.eps_d * 2 * BETA
                    deltaE_in_eV = newns.DeltaE * 2 * BETA

                    energies_dband.append(energy_in_eV)
                    energies_chemisorp.append(deltaE_in_eV)

                # The quantity that we want to plot
                data_scaling[eps_sigma].append( [ energies_dband, energies_chemisorp ] )

        # Plot the d-band energy scaling
        scaling_lines = []
        scaling_lines_error = []
        for i, eps_sigma in enumerate(data_scaling):

            # These are the d-band energies of the different adsorbates
            energies_dband_lower = data_scaling[eps_sigma][0][0]
            energies_dband_avg = data_scaling[eps_sigma][1][0]
            energies_dband_upper = data_scaling[eps_sigma][2][0]

            # Plot the scaling of the different adsorbates
            scaling_lines.append(data_scaling[eps_sigma][1][1])

            if i == 1:
                # Store different beta plots
                scaling_lines_error.append(data_scaling[eps_sigma][0][1])
                scaling_lines_error.append(data_scaling[eps_sigma][2][1])

        cax = ax.scatter(scaling_lines[0], scaling_lines[1], 
                    c=energies_dband_avg, cmap=plt.cm.copper, marker='*', 
                    label=r'$\beta^{\prime}=%1.1f$ eV'%BETA_PRIME[1])

        ax.scatter(scaling_lines[0], scaling_lines_error[0], 
                        c=energies_dband_avg, cmap=plt.cm.copper, marker='v',
                        label=r'$\beta^{\prime}=%1.1f$ eV'%BETA_PRIME[0])
        ax.scatter(scaling_lines[0], scaling_lines_error[1], 
                        c=energies_dband_avg, cmap=plt.cm.copper, marker='o', 
                        label=r'$\beta^{\prime}=%1.1f$ eV'%BETA_PRIME[2])

        if j == len(EPSILON_SIGMA_all) - 1:
            cbar = fige.colorbar(cax, ax=ax)
            cbar.ax.set_ylabel('d-band centre (eV)')
            # ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
            ax.legend(loc='best')

        ax.set_xlabel(r'$\Delta E(\epsilon_a = %1.1f)$ (eV)'%EPSILON_SIGMA[0])
        ax.set_ylabel(r'$\Delta E(\epsilon_a = %1.1f)$ (eV)'%EPSILON_SIGMA[1])
        # ax.legend(loc='best')

    fige.savefig('output/NewnsAnderson_scaling.png')





    




