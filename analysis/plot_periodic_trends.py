"""Plot the periodic trends of the Newns-Anderson paramaters."""
import numpy as np
import json
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()

# Define periodic table of elements
FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

if __name__ == '__main__':
    """Plot the periodic trends of the Newns-Anderson paramaters."""

    # Load the data
    with open('output/delta/fit_results.json', 'r') as f:
        json_data = json.load(f)

    # Make a separate plot for each adsorbate calculation
    for i, ads in enumerate(json_data):
        # Plot Beta, Beta_p and epsilon_d in this plot
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        for j, metals in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            for k, element in enumerate(metals):

                # Extract the data if it is available
                try:
                    beta = json_data[ads][element]['beta']
                    betap = json_data[ads][element]['beta_p']
                    epsilon_d = json_data[ads][element]['eps_d']
                except KeyError:
                    continue

                ax[0].plot(k, beta, 'o',  color=plt.cm.tab10(j))
                ax[1].plot(k, betap, 'o',  color=plt.cm.tab10(j))
                ax[2].plot(k, epsilon_d, 'o', color=plt.cm.tab10(j))

                ax[0].annotate(element, (k, beta), xytext=(k-0.3, beta), color=plt.cm.tab10(j), fontsize=14)
                ax[1].annotate(element, (k, betap), xytext=(k-0.3, betap), color=plt.cm.tab10(j), fontsize=14)
                ax[2].annotate(element, (k, epsilon_d), xytext=(k-0.3, epsilon_d), color=plt.cm.tab10(j), fontsize=14)

        # ax[0].set_xlabel('Element')
        ax[0].set_ylabel(r'$\beta$ (eV) ')
        # ax[1].set_xlabel('Element')
        ax[1].set_ylabel(r"$\beta'$ (eV)")
        # ax[2].set_xlabel('Element')
        ax[2].set_ylabel(r'$\epsilon_d$ (eV)')

        for d, element in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):
            ax[0].plot([], [], 'o', color=plt.cm.tab10(d), label=f'{d+1} row TM')
        ax[0].legend(loc='best', fontsize=14)
        # ax[0].set_xticks(np.arange(len(FIRST_ROW)))
        # ax[1].set_xticks(np.arange(len(FIRST_ROW)))
        # ax[2].set_xticks(np.arange(len(FIRST_ROW)))
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[2].set_xticks([])
        # ax[0].set_xticklabels(FIRST_ROW)
        # fig.suptitle('Periodic trends of the Newns-Anderson parameters')
        fig.savefig('output/delta/periodic_trends_adsorbate_{}.png'.format(ads))

