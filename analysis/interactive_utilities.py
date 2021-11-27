"""Functions for interactive plots of the Newns-Anderson model."""
import numpy as np
import matplotlib.pyplot as plt
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical


def interactive_newns_anderson_dos(Vak=1, eps_a=-1, eps_d=-5, width=4, 
                                   Delta0_mag=2, eps_sp_min=-15, eps_sp_max=15):
    """Function to interactively plot the Newns-Anderson model."""
    eps = np.linspace(-40, 40, 1000)

    hybridisation = NewnsAndersonNumerical(
        Vak = Vak,
        eps_a = eps_a,
        eps = eps,
        width = width,
        eps_d = eps_d,
        Delta0_mag = Delta0_mag,
        eps_sp_max = eps_sp_max,
        eps_sp_min = eps_sp_min,
    )
    # Run the calculation
    hybridisation.calculate_energy()
    hybridisation.calculate_occupancy()

    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Get the Newns-Anderson model outputs
    Delta = hybridisation.get_Delta_on_grid()
    Lambda = hybridisation.get_Lambda_on_grid()
    eps_diff = hybridisation.get_energy_diff_on_grid()
    pdos = hybridisation.get_dos_on_grid()

    # Plot the Newns-Anderson model
    ax[0].plot(eps, Delta, color='tab:red', lw=3) 
    ax[0].plot(eps, Lambda, color='tab:green', lw=3)
    # ax[0].plot(eps, eps_diff, color='tab:red', lw=3, alpha=0.25)
    ax[1].plot(eps, pdos, color='tab:blue', lw=3)

    plt.show()

