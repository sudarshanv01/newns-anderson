"""Plot variation of a single parameter for the Newns Anderson Model."""

import numpy as np
import matplotlib.pyplot as plt
from newns_anderson import NewnsAnderson
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':
    BAND_CENTER = np.linspace(-5, 5, 6)
    COUPLING_ELEMENT = 2
    fig, ax = plt.subplots(1, len(BAND_CENTER), figsize=(3*len(BAND_CENTER), 5), constrained_layout=True)

    for i, eps_d_i in enumerate(BAND_CENTER):
        parameters = {'eps_d': eps_d_i, 'V':COUPLING_ELEMENT}
        model = NewnsAnderson(**parameters)
        model.run()

        ax[i].plot(model.Delta, model.eps, 'k-', lw=3, label='$\Delta$')
        ax[i].axhline(eps_d_i, color='k', ls='--', label=r'$\epsilon_{d}$')
        ax[i].plot(model.Lambda, model.eps, '-', color='tab:blue', label=r'$\Lambda$')
        ax[i].plot((model.eps - model.eps_a)/model.V, model.eps, '-', color='tab:red', label='$\epsilon - \epsilon_{\mathregular{Fermi}}$')
        ax[i].axhline(model.eps_a, color='tab:green', ls='--', label=r'$\epsilon_{a}$')
        ax[i].plot(model.na, model.eps, '-', color='tab:green', lw=3, label='$n_a$')
        xlim = [-np.max(model.Delta), np.max(model.Delta)]
        ax[i].set_xlim(xlim)
        if i == 0:
            ax[i].set_ylabel('$\epsilon - \epsilon_{\mathregular{Fermi}}$ (eV)')
        if i == len(BAND_CENTER)-1:
            ax[i].legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)

    fig.savefig('output/news_anderson_single.png')

