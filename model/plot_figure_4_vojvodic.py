"""Recreate Figure 4 of Vojvodic et al. (2014)."""
import numpy as np
import matplotlib.pyplot as plt
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':
    """Recreate Figure 4 of Vojvodic et al. (2014), including the 
    additional plot of na changing along the same dimensions."""

    # Create a contour plot of the energy matrix
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Parameters to change delta
    widths = np.linspace(1, 12, 20)
    eps_ds = np.linspace(-6, 5.5, 20)
    EPS_A = -5
    EPS_RANGE = np.linspace(-15, 15, 1000,) 
    delta0 = 0
    Vak = 1

    energy_matrix = np.zeros((len(widths), len(eps_ds)))
    na_matrix = np.zeros((len(widths), len(eps_ds)))

    for i, width in enumerate(widths):
        for j, eps_d in enumerate(eps_ds):

            newns = NewnsAndersonNumerical(
                width = width,
                Vak = Vak, 
                eps_a = EPS_A,
                eps_d = eps_d,
                eps = EPS_RANGE,
                Delta0 = delta0, 
            )
            newns.calculate_energy()
            newns.calculate_occupancy()

            energy_matrix[i, j] = newns.get_energy()
            na_matrix[i, j] = newns.get_occupancy()

    # Plot the contour
    energy_matrix = energy_matrix.T
    na_matrix = na_matrix.T

    cax = ax[0].contourf(widths, eps_ds, energy_matrix, levels=100)
    cbar = fig.colorbar(cax, ax=ax[0])
    cax = ax[1].contourf(widths, eps_ds, na_matrix, levels=100)
    cbar = fig.colorbar(cax, ax=ax[1])
    # plot the contour lines
    ax[0].contour(widths, eps_ds, energy_matrix, levels=10, colors='k')
    ax[1].contour(widths, eps_ds, na_matrix, levels=10, colors='k')
    ax[0].set_xlabel(r'width (eV)')
    ax[0].set_ylabel(r'$\epsilon_d$ (eV)')
    ax[1].set_xlabel(r'width (eV)')
    ax[1].set_ylabel(r'$\epsilon_d$ (eV)')
    ax[0].set_title(r'Chemisorption Energy (eV)')
    ax[1].set_title(r'$n_a$ (e)')
    fig.savefig('output/figure_4_vojvodic.png', dpi=300)



