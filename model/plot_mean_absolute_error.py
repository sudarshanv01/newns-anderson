"""Plot the mean absolute error for a linear fit for different epsilon_a values."""
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
from scipy import stats
import json
get_plot_params()

if __name__ == '__main__':

    eps_a_1 = np.linspace(-6, 5, 50)
    eps_a_2 = np.linspace(-6, 5, 50)
    eps_d_all = np.linspace(-8, -2, 50)
    BETA = 2 # Metal d-band width
    BETA_P = 2
    EPSILON_RANGE = np.linspace(-20, 20, 500) # range of energies plot in dos
    FERMI_ENERGY = 0.0 # Fermi energy
    U = 0 # No Coulomb interaction

    mean_error = np.zeros((len(eps_a_1), len(eps_a_2)))

    for i in range(len(eps_a_1)):
        for j in range(len(eps_a_2)):
            energies_1 = []
            energies_2 = []
            for k, eps_d in enumerate(eps_d_all):
                # For the first adsorbate
                newns_1 = NewnsAndersonAnalytical(beta = BETA, 
                                                beta_p = BETA_P / 2 / BETA, 
                                                eps_d = eps_d,
                                                eps_a = eps_a_1[i],
                                                eps = EPSILON_RANGE,
                                                fermi_energy = FERMI_ENERGY,
                                                U = U)
                newns_1.self_consistent_calculation()
                energy_in_eV = newns_1.DeltaE * 2 * BETA 
                energies_1.append(energy_in_eV)
                # For the second adsorbate
                newns_2 = NewnsAndersonAnalytical(beta = BETA, 
                                                beta_p = BETA_P / 2 / BETA, 
                                                eps_d = eps_d,
                                                eps_a = eps_a_2[j],
                                                eps = EPSILON_RANGE,
                                                fermi_energy = FERMI_ENERGY,
                                                U = U)
                newns_2.self_consistent_calculation()
                energy_in_eV = newns_2.DeltaE * 2 * BETA 
                energies_2.append(energy_in_eV)

            # Fit the value of energies to the epsilon_d to check
            # if the fit is good and store the mean absolute error in mean error
            energies_epsd_fit = np.polyfit(energies_1, energies_2, 1)
            energies_epsd_fit = np.poly1d(energies_epsd_fit)
            energies_epsd_fit = energies_epsd_fit(energies_1)
            result = stats.linregress(energies_1, energies_2)
            mean_error[i, j] = result.rvalue**2
            # mean_error[i, j] =  #np.mean(np.abs(energies_2 - energies_epsd_fit))

    fig, ax = plt.subplots(1, 1, figsize=(8,7), constrained_layout=True) 

    # Save the data
    data = {}
    data['eps_a_1'] = list(eps_a_1)
    data['eps_a_2'] = list(eps_a_2)
    data['mean_error'] = mean_error.tolist()

    with open('output/mean_error.json', 'w') as f:
        json.dump(data, f)

    ax.set_title(r"$\beta=%1.1f \mathregular{eV}, \beta^\prime=%1.1f \mathregular{eV}$"%(BETA, BETA_P))
    CS = ax.contourf(eps_a_1, eps_a_2, mean_error, 100, cmap=plt.cm.RdBu)
    ax.set_ylabel(r'$\epsilon_{a1} - \epsilon_f$ (eV)')
    ax.set_xlabel(r'$\epsilon_{a2} - \epsilon_f$ (eV)')
    ax.annotate('CO, OH', xy=(2.5, -1), color='white')
    ax.annotate('CO$_2$, CO', xy=(1, 2.5), color='white')
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('R$^2$ values')
    fig.savefig('output/NewnsAnderson_mean_error.png')


