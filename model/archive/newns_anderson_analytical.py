
# ------ regular imports
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 
from pprint import pprint
import numpy as np
import click
import json
from scipy.signal import hilbert
from ase.db import connect
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from scipy.integrate import quad, simps
from scipy.optimize import curve_fit
from matplotlib import cm
# ------ specific imports
from get_plot_params import get_plot_params_Arial
get_plot_params_Arial()

def adsorbate_states(delta, hilbert, eps_a, eps, V=1):
    na =  1 / np.pi * ( delta ) / ( ( (eps - eps_a)/V**2 - hilbert )**2 + delta**2  ) 
    return na

def semi_elliptical_dos(a, b, c, energy):
    """Plotting semi-ellipse as Delta (state of the metal)
    Equation is of the form
    delta = a ( c - energy**2 / b ) ** 0.5
    """
    delta = []
    for E in energy:
        d = a * ( 1 -  (E-c)**2 / b) #** 1/2
        if d < 0:
            delta.append(0)
        else:
            delta.append(d)
    area = np.trapz(energy, delta)
    return np.array(delta)

def lorentz_dos(a, b1,energy):
    delta = 1 / np.pi * ( a / ( (energy - b1)**2 + a**2  ) )   
    return delta

def hybridization_energy(energy, e_a, Delta, Lambda, V):
    function = (2 / np.pi) * ( np.arctan( Delta / ( (energy - e_a)/V**2 - Lambda ) ) ) 
    dE_hyb = np.trapz(function, energy) - e_a
    return dE_hyb

def schematic():

    chosen_states = { 'H':[ 0.0 ]  }
    height = 0.4 # eV
    height2 = 0.2 # eV
    centers = np.linspace(-5, -1, 10)
    centers2 = np.linspace(-10,-5, 10)
    cmap = cm.get_cmap('viridis',len(centers))
    V = 1.5
    colors = ['tab:blue', 'tab:green', 'tab:orange']

    ## collect all hybridisation energies to plot against
    all_E = {}
    # figs, axs = plt.subplots(len(chosen_states), len(chosen_states), figsize=(20,18))
    figs, axs = plt.subplots(1, 6, figsize=(22,4), constrained_layout=True)

    for state in chosen_states:
        fig, ax = plt.subplots(1,len(centers), figsize=(10,2), constrained_layout=True)
        figo, axo = plt.subplots(1,len(chosen_states[state]),figsize=(5*len(chosen_states[state]),4),squeeze=False)
        figd, axd = plt.subplots(1,1, figsize=(6,4), constrained_layout=True)

        all_epsa = chosen_states[state]
        energies = np.linspace(-15,2,100)

        all_E[state] = []
        all_filling = {}

        for i, ea in enumerate(chosen_states[state]):
            all_filling[i] = []

        for i, center in enumerate(centers):
            ## chemisorption function
            delta = lorentz_dos(height, centers[i], energies.tolist()) + lorentz_dos(height2, centers2[i], energies.tolist()) 
            # delta = semi_elliptical_dos(height/4, 1**2, center, energies.tolist())
            ax[i].plot(delta, energies, color='tab:red')
            ax[i].fill_between(delta, energies, color='tab:red', alpha=0.25)
            bounds = ax[i].get_xbound()
            ## hilbert transform
            hilbert = np.imag(signal.hilbert(delta))

            E_hyb = 0.0
            for k, eps_a in enumerate(all_epsa):
                na = adsorbate_states(np.array(delta), hilbert, eps_a, energies, V)
                filled_indices = [a for a in range(len(energies)) if energies[a] < 0 ]
                filled_na = na[filled_indices]
                occupancy = np.trapz(filled_na, energies[filled_indices]) 
                E_hyb += hybridization_energy(energies[filled_indices], eps_a, delta[filled_indices], hilbert[filled_indices], V)
                ax[i].plot(na, energies, color=colors[k])
                ax[i].fill_between(na[filled_indices], energies[filled_indices],  color='k', alpha=0.25)
                ax[i].axhline(eps_a, color=colors[k], alpha=0.5, ls='--')
                all_filling[k].append(occupancy)
            all_E[state].append(E_hyb)

            ax[i].set_xticks([])
            ax[i].set_xlim(bounds)
            if i != 0:
                ax[i].set_yticks([])
            else:
                ax[i].set_ylabel(r'$\epsilon - \epsilon_{f}$ / eV')
        for k in all_filling:
            axo[0,k].plot((centers+centers2)/2, all_filling[k], 'o-', color=colors[k])
            

        for k, eps_a in enumerate(all_epsa):
            axo[0,k].plot([], [], 'o', color=colors[k], label='$\epsilon_a = %1.1f eV$'%eps_a)
            axo[0,k].set_ylabel(r'Filling')
            axo[0,k].set_xlabel(r'$\epsilon_{d}$ / eV')
        axd.set_ylabel(r'$E_{\mathrm{hyb}}$ / eV')
        axd.set_xlabel(r'$\epsilon_{d}$ / eV')
        axd.plot((centers+centers2)/2, all_E[state], 'o-', color='k')

        figd.savefig('output/%s_hybridisation.png'%state)
        plt.close(figd)
        fig.savefig('output/%s_dos.png'%state)
        figo.tight_layout()
        figo.savefig('output/%s_occupancy.png'%state)
        plt.close(figo)
    

if __name__ == "__main__":
    schematic()

