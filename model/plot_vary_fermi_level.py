
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()


if __name__ == '__main__':
    # Define the parameters
    BETA = 2  # interaction with metal atoms, eV
    METAL_ENERGIES = 0 # center of d-band, eV
    BETA_PRIME = [2, 4] # Interaction with the adsorbate, eV 
    ADSORBATE_ENERGIES = [-5, -2] # eV 
    EPSILON_RANGE = np.linspace(-10, 10 , 10000) # eV 
    PLOT_DOS = False # Plot the dos is the number is small
    FERMI_LEVEL = np.linspace(-5, 5, 20) # eV

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)

    for b, beta_p in enumerate(BETA_PRIME): 
        for d, eps_sigma in enumerate(ADSORBATE_ENERGIES):
            all_energies = []
            for e_fermi in FERMI_LEVEL:

                eps_sigma_corrected = eps_sigma - e_fermi
                eps_d_f = METAL_ENERGIES - e_fermi

                eps_range = EPSILON_RANGE 
                
                newns = NewnsAndersonAnalytical(beta = BETA, 
                                                beta_p = beta_p,
                                                eps_d = METAL_ENERGIES,
                                                eps_sigma = eps_sigma_corrected,
                                                eps = eps_range,
                                                fermi_energy=e_fermi/2/BETA)
                
                all_energies.append([eps_d_f, newns.DeltaE])
            
            all_energies = np.array(all_energies).T
            ax.plot(all_energies[0], all_energies[1], 'o-', label=r"$\epsilon_{\sigma}=%1.2f \beta'=%1.2f$"%(eps_sigma, beta_p)) 

    ax.set_xlabel(r'$\epsilon_{d} - \epsilon_{f}$ (eV)')
    ax.set_ylabel(r'$\Delta E$ ($2\beta$)')

    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    fig.savefig('output/NewnsAnderson_vary_fermi_level.png')
