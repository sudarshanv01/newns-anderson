"""Compare the Sc DOS from different codes."""

import json
import matplotlib.pyplot as plt
from plot_params import get_plot_params
from collections import defaultdict
import yaml
from ase.dft import get_distribution_moment
get_plot_params()

def plot_pdos(energies, pdos, label, color):
    """Plot the projected density of states."""
    center, second_moment = get_distribution_moment(energies, pdos, (1, 2)) 
    print(f'{label}: center = {center:.3f}, second moment = {second_moment:.3f}')
    ax.plot(energies, pdos, '-', label=label, color=color)
    ax.axvline(x=center, linestyle='--', alpha=0.5, color=color)

if __name__ == '__main__':
    """Plot the Sc dos from VASP and QE."""

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3), constrained_layout=True)
    # make color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # First from Quantum Espresso
    COMP_SETUP = [ 'PBE_SSSP_efficiency_tetrahedron_smearing_dos_scf',
                   'PBE_SSSP_efficiency_cold_smearing_0.1eV_dos_scf',
                   'PBE_SSSP_precision_gauss_smearing_0.1eV_dos_scf_small',
    ]
    labels = ['tetrahedron (QE-efficiency)', 'cold smearing (QE-efficiency)', 'gauss smearing (QE-precision)']
    for i, setup in enumerate(COMP_SETUP):
        with open(f'output/pdos_{setup}.json', 'r') as handle:
            pdos_data = json.load(handle)
        energies, pdos, _ = pdos_data['slab']['Sc']
        plot_pdos(energies, pdos, label=labels[i], color=colors[i])

    # Now plot the different VASP calculations
    with open(f'output/Sc_dos_vasp_surface_0.1eV_smearing.json', 'r') as handle:
        pdos_data = json.load(handle)
    energies = pdos_data['energies']
    pdos = pdos_data['pdos']
    plot_pdos(energies, pdos, label='gaussian smearing (VASP)', color=colors[i+1])

    ax.legend(loc='best', fontsize=8)
    ax.set_xlim([-10, 10])
    ax.set_yticks([])
    ax.set_ylabel('$d$-projected dos')
    ax.set_xlabel('$\epsilon - \epsilon_F$ (eV)')


    fig.savefig('output/Sc_dos_compare.png', dpi=300)