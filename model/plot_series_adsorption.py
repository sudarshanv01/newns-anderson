"""Plot the expected sequence of events with the density of states."""
import numpy as np
import matplotlib.pyplot as plt
from catchemi import NewnsAndersonNumerical
from plot_params import get_plot_params, get_plot_params_Arial
get_plot_params_Arial()
import string
from scipy import signal


if __name__ == '__main__':
    """Make a panel with the adsorbate density of states
    as it goes from vacuum like states to those when it is
    adsorbed on a particular metal."""

    delta0_value = 0.3 # eV 
    width = 1.5 # eV
    Vak = 2.0 # eV
    EPS_A = -5.0 # eV
    EPS_D = -2 # eV
    EPS_RANGE = np.linspace(-30, 10, 1000,) 

    fig, ax = plt.subplots(1, 3, figsize=(8, 4.5), constrained_layout=True, sharey=True)
    for a in ax:
        a.set_ylim(-15, 5)
        a.set_xticks([])
        a.axis('off')
        a.axhline(0, color='k', linestyle='-')

    newns = NewnsAndersonNumerical(
        width = width,
        Vak = 0.0, 
        eps_a = EPS_A,
        eps_d = EPS_D,
        eps = EPS_RANGE,
        Delta0_mag = delta0_value, 
        eps_sp_max = 10, 
        eps_sp_min = -30,
    )
    # First plot the adsorbate state as a delta function at the
    # chosen eps_a value
    ax[0].axhline(EPS_A, color='tab:green', lw=3)
    # Plot the density of states interacting with the sp states
    dos_sp = newns.get_dos_on_grid()
    ax[1].plot(dos_sp, EPS_RANGE, lw=3, color='tab:green', label=r'$\rho_{aa}$')
    ax[1].fill_between(dos_sp, EPS_RANGE, color='tab:green', alpha=0.2)
    Delta = newns.get_Delta_on_grid()
    ax[1].plot(Delta, EPS_RANGE, lw=3, color='tab:purple', label=r'$\Delta$', alpha=0.5)
    # Plot the density of states interacting with the sp and d-states
    newns.Vak = Vak
    dos_sp_d = newns.get_dos_on_grid()
    ax[2].plot(dos_sp_d, EPS_RANGE, lw=3, color='tab:green', label=r'$\rho_{aad}$')
    ax[2].fill_between(dos_sp_d, EPS_RANGE, alpha=0.2, color='tab:green', label=r'$\rho_{aad}$')

    # Plot the metal density of states 
    Delta = newns.get_Delta_on_grid()
    ax[2].plot(Delta / np.max(Delta), EPS_RANGE, lw=3, color='tab:purple', label=r'$\Delta$', alpha=0.5)

    fig.savefig('output/schematic_ads.png')

    


