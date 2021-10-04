"""Recreate Figure 4 of Vojvodic et al. (2014)."""
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonNumerical, NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()
if __name__ == '__main__':
    """Recreate Figure 4 of Vojvodic et al. (2014)."""
    # Create a contour plot of the energy matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    widths = np.linspace(0.2, 12, 100)
    eps_ds = np.linspace(-6, 5.5, 100)
    EPS_A = -5
    EPS_RANGE = np.linspace(-20, 20, 200000)

    energy_matrix = np.zeros((len(widths), len(eps_ds)))

    for i, width in enumerate(widths):
        for j, eps_d in enumerate(eps_ds):

            newns = NewnsAndersonNumerical(
                width = width,
                Vak = 1, 
                eps_a = EPS_A,
                eps_d = eps_d,
                eps = EPS_RANGE,
            )
            newns.calculate_energy()

            energy_matrix[i, j] = newns.DeltaE

    # Plot the contour
    energy_matrix = energy_matrix.T
    cax = ax.contourf(widths, eps_ds, energy_matrix, levels=100)
    cbar = fig.colorbar(cax, ax=ax)
    # plot the contour lines
    ax.contour(widths, eps_ds, energy_matrix, levels=10, colors='k')
    ax.set_xlabel(r'width')
    ax.set_ylabel(r'$\epsilon_d$')
    ax.set_title(r'Chemisorption Energy')
    fig.savefig('output/figure_4_vojvodic.png')



