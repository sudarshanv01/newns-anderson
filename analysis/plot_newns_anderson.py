"""Plots the results of LCAO calculations for the Newns Anderson Model."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import index
from aiida import orm
from NewnsAndersonLCAO import NewnsAndersonLCAO
from plot_params import get_plot_params
get_plot_params()
BaseGPAW = WorkflowFactory('ase.gpaw.base')


if __name__ == '__main__':

    # Query for the data of the group
    GROUP_NAME = 'fcc_111/6x6x4/c_adsorption/scf_calculation'
    TYPE_OF_CALC = BaseGPAW 
    POSSIBLE_ADSORBATE_INDEX = ['C', 'O', 'N', 'H']
    ADSORBATE_BASIS_INDEX = 13 # Last elements of the H, S matrix

    # Query for the data of the calculation
    qb = orm.QueryBuilder()
    qb.append(Group, filters={'label':GROUP_NAME}, tag='Group')
    qb.append(TYPE_OF_CALC, with_group='Group', )


    for results in qb.all(flat=True):

        # Define the figures
        figd, axd = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
        fige, axe = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)

        # Get the structure and get relevant information
        # from that data
        structure = results.inputs.structure
        ase_structure = structure.get_ase()
        adsorbate_index = [atom.index for atom in ase_structure if atom.symbol in POSSIBLE_ADSORBATE_INDEX]
        metal_index = [atom.index for atom in ase_structure if atom.symbol not in POSSIBLE_ADSORBATE_INDEX]
        metal_name = np.unique([atom.symbol for atom in ase_structure if atom.symbol not in POSSIBLE_ADSORBATE_INDEX])[0]

        # Get the relevant quantities for the 
        # Newns Anderson Model
        arrays = results.outputs.array
        H_skMM = arrays.get_array('H_skMM')
        S_kMM = arrays.get_array('S_kMM')
        energies_dos = arrays.get_array('energies_dos')
        weights_dos = arrays.get_array('weights_dos')
        eigenvalues = arrays.get_array('eigenvalues')
        fermi_energy = results.outputs.parameters.get_attribute('fermi_energy')

        # Choose the right index for the subdiagonalisation
        # of the Hamiltonian
        H_MM = H_skMM[0,0]
        S_MM = S_kMM[0]

        # Partitition the index
        all_index = np.arange(len(H_MM[0]))
        adsorbate_index = all_index[-ADSORBATE_BASIS_INDEX:]
        metal_index = all_index[:-ADSORBATE_BASIS_INDEX]

        # Range of values for which we want delta to be plotted
        delta_range = np.array([-6, 6]) + fermi_energy 
        # Decide if more padding with zero's is required
        PAD_RANGE = np.array([10, 10]) # Pad with zeros 10 eV on either side
        # Perform all the manipulations of the Newns-Anderson model
        newns = NewnsAndersonLCAO(H_MM, S_MM, adsorbate_index, metal_index,
                                    broadening_width=0.1, cutoff_range=delta_range,
                                    pad_range=PAD_RANGE)
        # Plot the relevant quantities
        for i, eps_a in enumerate(newns.eigenval_ads):
            figa, axa = plt.subplots(1, 1, figsize=(8,4), constrained_layout=True)
            axa.axvline(fermi_energy, color='black', linestyle='--', label='Fermi Energy')
            axa.plot(newns.eigenval_metal, newns.delta[i], label='Delta', lw=3)
            axa.plot(newns.eigenval_metal, newns.Lambda[i], label='Lambda', lw=3)
            figa.savefig('output/delta/{}_delta_{}.png'.format(metal_name, np.round(eps_a,2)))
            plt.close(figa)