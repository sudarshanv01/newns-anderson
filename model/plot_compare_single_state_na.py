"""Plot the single state binding energy and the NA one."""
import numpy as np
import matplotlib.pyplot as plt
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical
from plot_params import get_plot_params
import string
get_plot_params()

def get_single_state_energy(eps_d, eps_a, V):
    """Get the binding energy from the 2 state model."""
    DeltaE = (eps_a - eps_d)**2  + 4*V**2
    DeltaE = -1 * np.sqrt(DeltaE)
    return DeltaE

def get_hybridised_states(eps_d, eps_a, V):
    """Get the bonding and anti-bonding state from the 
    2 state model."""
    anti_bonding = 0.5 * ( eps_d + eps_a ) + 0.5 * np.sqrt( (eps_d - eps_a)**2 + 4*V**2 )
    bonding = 0.5 * ( eps_d + eps_a ) - 0.5 * np.sqrt( (eps_d - eps_a)**2 + 4*V**2 )
    return bonding, anti_bonding

if __name__ == '__main__':
    """Compare the Newns-Anderson binding energy against the single state
    one in terms of variation of the d-band centre."""

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    fige, axe = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    # Parameters to change delta
    widths = [1, 2, 4, 6]
    eps_ds = np.linspace(-4, -1, 100)
    EPS_A = -3
    EPS_RANGE = np.linspace(-20, 20, 1000,) 
    delta0 = 0
    Vak = 1 

    # Get a cycle of colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Calculate the single state energy
    single_states = np.zeros((len(eps_ds),2))
    two_level_energies = np.zeros(len(eps_ds))

    for k, eps_d in enumerate(eps_ds):
        single_states[k,:] = get_hybridised_states(
            eps_d = eps_d,
            eps_a = EPS_A,
            V = Vak)
        # two_level_energies[k] = get_single_state_energy(
        #     eps_d = eps_d,
        #     eps_a = EPS_A,
        #     V = Vak)
        bonding, anti_bonding = single_states[k,:]
        two_level_energies[k] = bonding + anti_bonding - EPS_A - eps_d

    single_states = single_states.T
    ax[0].plot(eps_ds, single_states[0], lw=3, label='2-level', color='k')
    ax[1].plot(eps_ds, single_states[1], lw=3, label='2-level', color='k')
    axe.plot(eps_ds, two_level_energies, lw=3, label='2-level', color='k')

    for i, width in enumerate(widths):
        state_energy = np.zeros((len(eps_ds), 3))
        hyb_energies = np.zeros(len(eps_ds))

        for j, eps_d in enumerate(eps_ds):

            newns = NewnsAndersonNumerical(
                width = width,
                Vak = Vak, 
                eps_a = EPS_A,
                eps_d = eps_d,
                eps = EPS_RANGE,
                Delta0_mag = delta0, 
            )
            newns.calculate_energy()
            # newns.calculate_occupancy()

            # Get the energies of the states
            poles = newns.poles
            state_energy[j,:] = poles
            print(poles)

            # Get the hybridisation energies
            hyb_energies[j] = newns.get_energy()

        state_energy = state_energy.T
        ax[0].plot(eps_ds, state_energy[0], lw=3, label=f'width = {width}eV', color=colors[i])
        ax[1].plot(eps_ds, state_energy[2], lw=3, label=f'width = {width}eV', color=colors[i])
        axe.plot(eps_ds, hyb_energies, lw=3, label=f'width = {width}eV', color=colors[i])
    
    # Plot the single state energy
    for a in ax:
        a.set_xlabel(r'$\epsilon_d$')
    ax[0].set_ylabel(r'$\epsilon_{\mathregular{bonding}}$ (eV)')
    ax[1].set_ylabel(r'$\epsilon_{\mathregular{anti-bonding}}$ (eV)')
    ax[0].legend(loc='best', fontsize=12)

    axe.set_xlabel(r'$\epsilon_d$')
    axe.set_ylabel(r'$\Delta E$ (eV)')

    for i, ax in enumerate(ax):
        ax.text(0.1, 0.8, string.ascii_lowercase[i] +')', transform=ax.transAxes,
            fontsize=16,  va='top', ha='right')

    fig.savefig('output/plot_compare_single_state_na.png')
    fige.savefig('output/plot_compare_single_state_na_energy.png')