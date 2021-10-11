"""Plot scaling of the Newns Anderson Model based on adsorbates of two eps_a."""

import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonNumerical, NewnsAndersonAnalytical
import yaml
from plot_params import get_plot_params
get_plot_params()
if __name__ == '__main__':
    """Plot the scaling line based on the Newns-Anderson model."""

    # Scale these values
    EPS_A_VALUES = [ -1, -5 ]

    # All the parameters for scaling
    WIDTHS = [4, 5, 6]
    K_VALUE =  0.0
    EPS_RANGE = np.linspace(-20, 20, 200000)
    eps_ds = np.linspace(-6, 0, 40)
    Vak_VALUES = [1, 2, 3]
    Delta_0 = [2, 2, 2]

    # Figure for scaling
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    # Generate a sequence of linestyles
    linestyles = ['-', '--', '-.', ':']

    # Generate a sequence of markers
    markers = ['o', 'v', 's', 'p', '*', 'h', 'H', 'D', 'd']

    for i, Vak in enumerate(Vak_VALUES):
        energies_epsa = []
        for eps_a in EPS_A_VALUES:
            energies_epsd = []
            for j, eps_d in enumerate(eps_ds):
                # Run the Newns calculation
                newns = NewnsAndersonNumerical(
                    width = WIDTHS[i],
                    Vak = Vak, 
                    eps_a = eps_a,
                    eps_d = eps_d,
                    eps = EPS_RANGE,
                    k = Delta_0[i],
                )
                newns.calculate_energy()
                energies_epsd.append( newns.DeltaE )

            energies_epsa.append( energies_epsd )
        
        cax = ax.scatter(energies_epsa[0], energies_epsa[1], c=eps_ds, 
                marker=markers[i], label=r'$V_{ak}$' + f' $= {Vak}$ eV, ' + r'$w_{d}$' + f' $= {WIDTHS[i]}$ eV, '
                + r'$\Delta_0$' + f' $= {Delta_0[i]}$ eV', 
                cmap='coolwarm') 

    # Also plot the transition metals
    # Load data from vojvodic_parameters.yaml
    with open('vojvodic_parameters.yaml', 'r') as f:
        vojvodic_parameters = yaml.safe_load(f)
    for metal in vojvodic_parameters['epsd']:
        energies = []
        for eps_a in EPS_A_VALUES:
            eps_d = vojvodic_parameters['epsd'][metal]
            second_moment = vojvodic_parameters['mc'][metal]
            Vaksq = vojvodic_parameters['Vaksq'][metal]
            Vak = np.sqrt(Vaksq)
            # Get the width based on the second moment
            width = 4 * np.sqrt(second_moment)

            # Get the energy from the Newns Anderson model
            newns = NewnsAndersonNumerical(
                width = width,
                Vak = Vak, 
                eps_a = eps_a,
                eps_d = eps_d,
                eps = EPS_RANGE,
                k = Delta_0[-1], 
            )
            newns.calculate_energy()
            print(f'Metal: {metal} has a W/Vak ratio: {width/Vak}')

            energy = newns.DeltaE
            energies.append(energy)
        ax.plot(energies[0], energies[1], 'o', color='k')
        # Annotate the metal name at this point
        # ax.annotate(metal, xy=(energies[0], energies[1]))

    ax.set_ylabel('$\Delta E (\epsilon_a=%1.1f$) / eV'%EPS_A_VALUES[1],)
    ax.set_xlabel('$\Delta E (\epsilon_a=%1.1f$) / eV'%EPS_A_VALUES[0],)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.set_ylabel('$\epsilon_d$ (eV)')
    ax.legend(loc='best', fontsize=14)

    fig.savefig('output/NewnsAndersonNumerical_scaling.png')