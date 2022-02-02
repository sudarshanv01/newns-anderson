"""Compare the analytical and numerical solution of the Newns-Anderson model."""
import numpy as np
import matplotlib.pyplot as plt
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical
from plot_params import get_plot_params
get_plot_params()
import string

# def newns_anderson_analytical(Vak, width, eps_a):
#     """Get the analytical Newns Anderson energy based on 
#     the half-filled band."""
#     eps_a = eps_a / width
#     prefac = np.pi * Vak**2

#     deltaE = prefac * (1 - 2*prefac - eps_a**2)**0.5
#     deltaE *= np.log((1 - prefac + (1 - 2*prefac - eps_a**2)**0.5) * (prefac**2 + eps_a**2)**(-0.5)) 
#     deltaE += eps_a * (1 - prefac) + np.arctan2(eps_a,prefac)
#     deltaE *= (np.pi * ( 2 * prefac - 1) )**-1
#     deltaE += 0.5 * eps_a

#     return deltaE

if __name__ == '__main__':
    """Plot the numerical solution of the Newns-Anderson model with different
    Delta0 values and compare it to the analytical solution where Delta0 = 0."""

    delta0_values = np.linspace(0, 1, 10)
    width = 2.0 # eV
    Vak = 1.0 # eV
    EPS_A = -5.0 # eV
    EPS_D = -2.0 # eV
    EPS_RANGE = np.linspace(-20, 20, 1000,) 

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Iterate over Delta0 values 
    energies = []
    for delta0 in delta0_values:
        newns = NewnsAndersonNumerical(
            width = width,
            Vak = Vak, 
            eps_a = EPS_A,
            eps_d = EPS_D,
            eps = EPS_RANGE,
            Delta0_mag = delta0, 
            eps_sp_max = 15, 
            eps_sp_min = -15,
        )
        newns.calculate_energy()
        newns.calculate_occupancy()

        energies.append(newns.get_energy())

    # Plot the numerical solution
    ax[0].plot(delta0_values, energies, label='Numerical', lw=3)    
    ax[0].set_ylabel(r'$\Delta E_{\rm hyb}$ (eV)')
    ax[0].set_xlabel(r'$\Delta_0$ (eV)')

    # Plot the density of states for the last Delta0 value
    Delta = newns.get_Delta_on_grid()
    Lambda = newns.get_Lambda_on_grid()
    energy_diff = newns.get_energy_diff_on_grid()
    dos = newns.get_dos_on_grid()

    ax[1].plot(EPS_RANGE, Delta, lw=3, color='tab:red', label=r'$\Delta$')
    ax[1].plot(EPS_RANGE, Lambda, lw=3, color='tab:red', alpha=0.5, label=r'$\Lambda$')
    ax[1].plot(EPS_RANGE, energy_diff, ls='--', color='tab:red', alpha=0.5, label=r'$\epsilon - \epsilon_a$')
    ax[1].plot(EPS_RANGE, dos, lw=3, color='tab:green', label=r'$\rho_{aa}$')
    ax[1].set_ylim([-np.max(Delta)+delta0, np.max(Delta)+delta0])
    ax[1].set_xlabel(r'$\epsilon - \epsilon_f$ (eV)')
    ax[1].set_yticks([])
    ax[1].legend(loc='best', fontsize=12)

    for i, ax in enumerate(ax):
        ax.text(0.1, 0.1, string.ascii_lowercase[i] +')', transform=ax.transAxes,
            fontsize=16,  va='top', ha='right')

    fig.savefig('output/plot_compare_analytical_Delta0.png')

    


