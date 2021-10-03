"""Recreate Figure 4 of Vojvodic et al. (2014)."""
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonNumerical, NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()
if __name__ == '__main__':
    """Recreate Figure 4 of Vojvodic et al. (2014)."""
    # Create a contour plot of the energy matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    widths = np.linspace(0, 12, 100)
    eps_ds = np.linspace(-6, 5.5, 100)
    EPS_A = -5
    EPS_RANGE = np.linspace(-35, 35, 10000)

    energy_matrix = np.zeros((len(widths), len(eps_ds)))

    for i, width in enumerate(widths):
        for j, eps_d in enumerate(eps_ds):

            # newns = NewnsAndersonAnalytical(
            #     beta_p = 1,
            #     eps_a = EPS_A,
            #     beta = width,
            #     eps_d = eps_d,
            #     eps = EPS_RANGE,
            #     fermi_energy = 0,
            #     U = 0., 
            # )
            # newns.self_consistent_calculation()

            newns = NewnsAndersonNumerical(
                width = width,
                Vak = 1, 
                eps_a = EPS_A,
                eps_d = eps_d,
                eps = EPS_RANGE,
            )
            newns.calculate_energy()

            if i == 20 and j == 90:
                ax[0].plot(newns.eps, newns.Delta, 'k-', label='$\Delta$')
                ax[0].plot(newns.eps, newns.Lambda, 'k--', label='$\Lambda$')
                ax[0].annotate(r'$w_d = %1.2f$'%newns.width, xy=(0.6,0.8), xycoords='axes fraction')

            energy_matrix[i, j] = newns.DeltaE

    ax[0].set_xlabel('$\epsilon_d$')
    ax[0].set_ylabel('$\Delta, \Lambda$')

    # Plot the contour
    energy_matrix = energy_matrix.T
    cax = ax[1].contourf(widths, eps_ds, energy_matrix, levels=100)
    cbar = fig.colorbar(cax, ax=ax[1])
    # plot the contour lines
    ax[1].contour(widths, eps_ds, energy_matrix, levels=20, colors='k')
    ax[1].set_xlabel(r'width')
    ax[1].set_ylabel(r'$\epsilon_d$')
    ax[1].set_title(r'$\Delta E$')
    fig.savefig('output/figure_4_vojvodic.png')



