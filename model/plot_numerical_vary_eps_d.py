"""Plot the variation of energy with the d-band center based on the numerical class of the 
Newns-Anderson model."""
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonNumerical, NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()
if __name__ == '__main__':
    """Plot the variation of the chemisorption energy against the d-band center."""

    # Create a contour plot of the energy matrix
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    WIDTHS = [ 2, 4, 6 ]
    K_VALUES = [ 2 ]
    eps_ds = np.linspace(-6, 5.5, 100)
    EPS_A = -1
    EPS_RANGE = np.linspace(-20, 20, 200000)
    Vak = 1

    # Create a range of sequential colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(WIDTHS)))
    # Generate a sequence of linestyles
    linestyles = ['-', '--', '-.', ':']

    for i, width in enumerate(WIDTHS):
        for k_index, k in enumerate(K_VALUES):
            energies = []
            for j, eps_d in enumerate(eps_ds):
                newns = NewnsAndersonNumerical(
                    width = width,
                    Vak = Vak, 
                    eps_a = EPS_A,
                    eps_d = eps_d,
                    eps = EPS_RANGE,
                    k = k,
                )
                newns.calculate_energy()
                
                energies.append( newns.DeltaE )
            if i == 1:
                ax[0].plot(newns.eps, newns.Delta, label=f'$\Delta_0 = {k}$ eV', color=colors[k_index])
                ax[0].plot(newns.eps, newns.Lambda, color=colors[k_index], ls=linestyles[k_index], alpha=0.5)

            ax[1].plot(eps_ds, energies,
                        label=f'width: {width} eV, $\Delta_0$: {k} eV',
                        lw=2, color=colors[i], ls=linestyles[k_index])

    # for a in ax:
        # a.legend(loc='best', fontsize=12)
    ax[0].legend(loc='best', fontsize=12)
    ax[1].legend(bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=14)

    ax[0].set_xlabel('$\epsilon$ (eV)')
    ax[0].set_ylabel('$\Delta$ (eV)') 
    ax[1].set_xlabel('$\epsilon_d$ (eV)')
    ax[1].set_ylabel('$\Delta E_{\mathregular{d-hyb}}$ (eV)') 
        # - \Delta E_{\mathregular{d-hyb}}(\epsilon_d=%1.1f)$ (eV)'%eps_ds[0], fontsize=14)
    fig.savefig('output/NewnsAndersonNumerical_vary_epsd.png')
