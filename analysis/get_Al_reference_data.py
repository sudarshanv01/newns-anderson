"""Get reference data for Al."""
import aiida
from collections import defaultdict
from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from pprint import pprint
from plot_params import get_plot_params
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
get_plot_params()

# --- aiida imports
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
DosWorkflow = WorkflowFactory('quantumespresso.pdos')

def get_density_of_states_for_node(node, element, angular_momentum=2,
                                   fermi_energy=0.0, position=None):
    """Get the projected density of states for all the atoms."""
    # get the projection data
    projwfc_data = node.outputs.projwfc
    pdos_data = projwfc_data.projections
    pdos_data = pdos_data.get_pdos(kind_name=element, angular_momentum=angular_momentum)

    # get the energy and reference it to the Fermi energy
    energy = pdos_data[0][-1] - fermi_energy

    # Sum up the pdos into one list
    pdos = np.zeros(len(energy))
    for index, values in enumerate(pdos_data):
        # check if the pdos data is for the atom we are looking for
        if position is not None:
            orb_dict = pdos_data[index][0].get_orbital_dict()
            if all(orb_dict['position'] == position):
                pdos += np.array(pdos_data[index][1])
        else:
            pdos += np.array(pdos_data[index][1])
    
    return list(energy), list(pdos) 

@dataclass
class DataFromDFT:
    groupname: str

    def __post_init__(self):
        self.pdos = defaultdict( lambda: defaultdict(dict) )
        self.get_data()
    
    def get_scf_energies(self, adsorbate, type_calc='dos'):
        """From the dos node get the scf calculation and return the energy."""
        if type_calc == 'dos':
            scf_node = self.node.get_outgoing(PwBaseWorkChain).get_node_by_label('scf')
            nscf_node = self.node.get_outgoing(PwBaseWorkChain).get_node_by_label('nscf')
            # Return the self-consistent energy
            energy = scf_node.outputs.output_parameters.get_attribute('energy')
            # Return also the Fermi energy
            fermi_energy = nscf_node.outputs.output_parameters.get_attribute('fermi_energy')

            # Get information about the structure
            structure = scf_node.inputs.pw.structure
            ase_structure = structure.get_ase()
            metal = np.unique(ase_structure.get_chemical_symbols())
            metal = metal[~np.isin(metal, self.adsorbate)]

            return metal, energy, fermi_energy, ase_structure 
        
        elif type_calc == 'relax':
            energy = self.node.outputs.output_parameters.get_attribute('energy')

            # Get information about structure
            try:
                structure = self.node.outputs.output_structure
            except aiida.common.exceptions.NotExistentAttributeError:
                structure = self.node.inputs.pw.structure
            ase_structure = structure.get_ase()
            metal = np.unique(ase_structure.get_chemical_symbols())
            metal = metal[~np.isin(metal, self.adsorbates)]

            return metal, energy, None, ase_structure 

    def get_data(self):
        """Get the raw energy from the SCF calculation."""
        type_of_calc = DosWorkflow 

        # Create the query for this adsorbate
        qb = QueryBuilder()
        qb.append(Group, filters={'label':self.groupname}, tag='Group')
        qb.append(type_of_calc, with_group='Group', tag='calctype')

        # Get information from the extras of the nodes
        for node in qb.all(flat=True):
            if not node.is_finished_ok:
                continue
            
            self.metal = node.get_extra('metal')
            facets = node.get_extra('facets')
            self.adsorbate = node.get_extra('adsorbate')
            
            # Get the energy
            self.node = node
            metal, energy, fermi_energy, ase_structure = self.get_scf_energies(self.adsorbate, type_calc='dos')

            # For the p-states
            sum_dos = []
            for adsorbate in list(self.adsorbate):
                print(adsorbate)
                energies, pdos_s = get_density_of_states_for_node(node, adsorbate,
                                                                    angular_momentum=0,
                                                                    fermi_energy=fermi_energy)
                sum_dos.append(pdos_s)
                try:
                    energies, pdos_p = get_density_of_states_for_node(node, adsorbate,
                                                                        angular_momentum=1,
                                                                        fermi_energy=fermi_energy)
                    sum_dos.append(pdos_p)
                except IndexError:
                    print('No p-states')
                    pass

            # Sum over the pdos over the y-axis
            sum_dos = np.sum(sum_dos, axis=0)
            assert len(sum_dos) == len(energies)
            pdos_to_store = [ energies, list(sum_dos) ]
                
            self.pdos[self.adsorbate][self.metal] = pdos_to_store
                
if __name__ == '__main__':
    """Get the d-band center, band width, chemisorption energy from a DFT calculation."""
    GROUPNAME = 'PBE/SSSP_precision/gauss_smearing_0.1eV/sampled/dos/Al_reference/adsorbates'
    LABEL = 'reference_Al'

    # Get the data from the DFT calculations done with AiiDA
    data = DataFromDFT(GROUPNAME) 

    # Save the pdos
    with open(f'output/pdos_{LABEL}.json', 'w') as handle:
        json.dump(data.pdos, handle, indent=4)