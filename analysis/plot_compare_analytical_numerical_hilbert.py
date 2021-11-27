"""Compare the analytical Hilbert transform with the numerical Hilbert transform."""
import numpy as np
from norskov_newns_anderson.NewnsAnderson import NewnsAndersonNumerical, NorskovNewnsAnderson 
from scipy import signal
import matplotlib.pyplot as plt
from plot_params import get_plot_params
get_plot_params()
if __name__ == "__main__":
    """Compare the Analytical Hilbert transform from the 
    NewnsAndersonNumerical class with the numerical Hilbert
    transform coming from signal.hilbert."""
    # Define the parameters
    Delta0 = 5.0
    Vak = 1.0
    eps_a = -1.0 # Dummy value
    eps_d = -2.0

    # Comparison of the analytical and numerical Hilbert transform
    fig, ax = plt.subplots(1, 1, figsize=(8,6), constrained_layout=True)

    # Run the class 
    hybridisation = NewnsAndersonNumerical(
        Vak = 3, 
        eps_a = eps_a, 
        eps_d = eps_d,
        width = 2,
        eps = np.linspace(-15, 15, 1000),
        Delta0 = Delta0,
    )

    hybridisation.calculate_energy()

    # Get the analytical quantities
    Delta_na = hybridisation.get_Delta_on_grid()
    Delta_na += hybridisation.Delta0
    Lambda_na = hybridisation.get_Lambda_on_grid()

    # Get the numerical quantities
    num_Lambda = np.imag(signal.hilbert(Delta_na))

    # Plot the two quantities
    ax.plot(hybridisation.eps, Delta_na, lw=3, label=r"$\Delta_{\mathrm{newns}}$")
    ax.plot(hybridisation.eps, Lambda_na, lw=3, label=r"$\Lambda_{\mathrm{newns}}$")
    ax.plot(hybridisation.eps, num_Lambda, lw=3, label=r"$\Lambda_{\mathrm{num}}$")

    # Set the labels
    ax.set_xlabel(r"$\epsilon$")
    ax.legend(loc='best')

    fig.savefig('output/comparison_of_lambda.png')
    