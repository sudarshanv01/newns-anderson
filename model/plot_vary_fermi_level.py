
import numpy as np
import matplotlib.pyplot as plt
from NewnsAnderson import NewnsAndersonAnalytical
from plot_params import get_plot_params
get_plot_params()

def plot_dos(ax):
    """Plot the density of states on an axis."""
    ax.axvline( newns.eps_d - newns.fermi_energy , color='k', ls='--' )
    ax.axvline( newns.eps_sigma - newns.fermi_energy , color='b', ls='--')
    ax.axvline( newns.lower_band_edge - newns.fermi_energy , ls='-.', color='tab:grey')
    ax.plot( newns.lower_band_edge - newns.fermi_energy, newns.Lambda_at_band_edge, '*', color='tab:green')
    ax.plot( newns.lower_band_edge - newns.fermi_energy, newns.Delta_at_band_edge, '*', color='tab:green')
    ax.axvline( newns.root_positive - newns.fermi_energy, ls='-.', color='tab:olive')
    ax.axvline( newns.root_negative - newns.fermi_energy, ls='-.', color='tab:olive')
    ax.plot( newns.eps_l_sigma - newns.fermi_energy, 0, '*', ms=14, color='tab:red')

    # All quantities plotted in units of 2beta
    ax.plot( newns.eps - newns.fermi_energy , newns.Delta, label = r'$\Delta$', lw=3)
    ax.plot( newns.eps - newns.fermi_energy , newns.Lambda, label = r'$\Lambda$', lw=3)
    ax.plot( newns.eps - newns.fermi_energy , newns.eps - newns.eps_sigma, label = r'$\epsilon$' )

    # Plot the density of states of the adsorbate
    ax.fill_between( newns.eps - newns.fermi_energy , newns.rho_aa, color='tab:red', label='$\rho_{aa}$')

    ax.set_yticks([])
    ax.set_xlabel('$\epsilon - \epsilon_{f}$')
    ax.set_xlim(np.min(eps_range)/2/BETA, np.max(eps_range)/2/BETA)
    # Annotate quantities
    ax.annotate( r"$\beta' = %.1f$" % (beta_p / 2 / BETA),
                        xy = (0.01, 0.9),
                        xycoords='axes fraction',
                        horizontalalignment='left',
                        verticalalignment='top' )

    if newns.has_localised_occupied_state:
        ax.annotate( r"Localised state", 
                        xy=(0.9,0.01),
                        xycoords='axes fraction',
                        horizontalalignment='right',
                        verticalalignment='bottom' )

if __name__ == '__main__':
    # Define the parameters
    BETA = 2  # interaction with metal atoms, eV
    METAL_ENERGIES = 0 # center of d-band, eV
    BETA_PRIME = [1, 2, 4] # Interaction with the adsorbate, eV 
    ADSORBATE_ENERGIES = [-5, -2] # eV 
    EPSILON_RANGE = np.linspace(-10, 10, 10000) # eV 
    PLOT_DOS = False # Plot the dos is the number is small
    FERMI_LEVEL = np.linspace(-5, 10, 20) # eV

    # This is the energy range that is with respect to 
    # d-band energy 
    lower_eps_range = np.min(EPSILON_RANGE)+np.min(FERMI_LEVEL)
    higher_eps_range = np.max(EPSILON_RANGE)+np.max(FERMI_LEVEL)
    eps_range = np.linspace(lower_eps_range, higher_eps_range, len(EPSILON_RANGE))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)

    for b, beta_p in enumerate(BETA_PRIME): 
        if PLOT_DOS:
            figd, axd = plt.subplots(len(ADSORBATE_ENERGIES), len(FERMI_LEVEL), 
                                        figsize=(5*len(FERMI_LEVEL), 4*len(ADSORBATE_ENERGIES)),
                                        constrained_layout=True)

        for d, eps_sigma in enumerate(ADSORBATE_ENERGIES):
            all_energies = []
            # Since we define the metal states as being at 0
            # We can use the fermi level as an alias to shift around the d-band center
            for f, e_fermi in enumerate(FERMI_LEVEL):
                # We have to adjust the Fermi energy to make sure that 
                # it stays constant as an absolute value
                eps_sigma_corrected = eps_sigma + e_fermi / 2  / BETA
                # The d-band energy will stay 0
                eps_d_f = METAL_ENERGIES

                
                newns = NewnsAndersonAnalytical(beta = BETA, 
                                                beta_p = beta_p / 2 / BETA,
                                                eps_d = METAL_ENERGIES,
                                                eps_sigma = eps_sigma_corrected,
                                                eps = eps_range,
                                                fermi_energy=e_fermi / 2 / BETA,
                                                use_analytical_roots=True)
                
                all_energies.append([-e_fermi, newns.DeltaE-e_fermi])

                if newns.has_localised_occupied_state:
                    print('Localised state')
                if PLOT_DOS:
                    plot_dos(axd[d,f])

            if PLOT_DOS:
                figd.savefig(f'output/NewnsAnderson_vary_fermi_level_vary_betap_{beta_p}.png')
            
            all_energies = np.array(all_energies).T
            ax.plot(all_energies[0], all_energies[1], 'o-', label=r"$\epsilon_{\sigma}=%1.2f \beta'=%1.2f$"%(eps_sigma, beta_p)) 

    ax.set_xlabel(r'$\epsilon_{d} - \epsilon_{f}$ (eV)')
    ax.set_ylabel(r'$\Delta E$ ($2\beta$)')

    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    fig.savefig('output/NewnsAnderson_vary_fermi_level.png')

