"""Get data needed for the Newns-Anderson model from the DFT calculation."""
from collections import defaultdict
from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from plot_params import get_plot_params
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
get_plot_params()
# --- aiida imports
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
DosWorkflow = WorkflowFactory('quantumespresso.pdos')

def get_density_of_states_for_node(node, element, angular_momentum=2, fermi_energy = 0.0):
    """Get the projected density of states for each spin for the dopant atom."""
    projwfc_data = node.outputs.projwfc
    pdos_data = projwfc_data.projections
    pdos_data = pdos_data.get_pdos(kind_name=element, angular_momentum=angular_momentum)

    # get the energy and reference it to the Fermi energy
    energy = pdos_data[0][-1] - fermi_energy

    # Sum up the pdos into one list
    pdos = np.zeros(len(energy))
    for index, values in enumerate(pdos_data):
        pdos += np.array(pdos_data[index][1]) 
    
    return list(energy), list(pdos) 

@dataclass
class DataFromDFT:
    base_group_name: str
    adsorbates: list
    references: dict

    def __post_init__(self):
        """Get the energies, d-band center and the width."""
        self.raw_energies = defaultdict( lambda: defaultdict(dict) )
        self.adsorption_energies = defaultdict ( lambda: defaultdict(float) )
        self.pdos = defaultdict( lambda: defaultdict(dict) )
        self.d_band_center = defaultdict(list)
        self.d_band_width = defaultdict(list)
        # Add the slab to the adsorbate
        if 'slab' not in self.adsorbates:
            self.adsorbates.append('slab')
        
        self.get_raw_energies()
        self.get_DFT_chemisorption_energy()
    
    def get_scf_energies(self):
        """From the dos node get the scf calculation and return the energy."""
        scf_node = self.node.get_outgoing(PwBaseWorkChain).get_node_by_label('scf')
        nscf_node = self.node.get_outgoing(PwBaseWorkChain).get_node_by_label('nscf')

        # Get information about the structure
        structure = scf_node.inputs.pw.structure
        ase_structure = structure.get_ase()
        metal = np.unique(ase_structure.get_chemical_symbols())
        metal = metal[~np.isin(metal, ['C', 'O'])]

        # Return the self-consistent energy
        energy = scf_node.outputs.output_parameters.get_attribute('energy')
        # Return also the Fermi energy
        fermi_energy = nscf_node.outputs.output_parameters.get_attribute('fermi_energy')

        return metal, energy, fermi_energy 

    def get_raw_energies(self):
        """Get the raw energy from the SCF calculation."""
        for adsorbate in self.adsorbates:
            groupname = f"{self.base_group_name}/{adsorbate}"
            # Create the query for this adsorbate
            qb = QueryBuilder()
            qb.append(Group, filters={'label':groupname}, tag='Group')
            qb.append(DosWorkflow, with_group='Group', tag='calctype')

            for node in qb.all(flat=True):

                if not node.is_finished_ok:
                    continue

                # Get the energy
                self.node = node
                metal, energy, fermi_energy = self.get_scf_energies()

                print(f"{adsorbate} {metal} ")
                assert len(metal) == 1
                self.raw_energies[adsorbate][metal[0]] = energy 


                if adsorbate == 'slab':
                    print(fermi_energy)
                    pdos_to_store = get_density_of_states_for_node(node, metal, angular_momentum=2, fermi_energy=fermi_energy)
                    self.pdos[adsorbate][metal[0]] = pdos_to_store
                
    
    def get_DFT_chemisorption_energy(self):
        """Get the DFT chemisorption energy by subtracting the formation energies."""
        # Remove slab from self.adsorbates
        self.adsorbates.remove('slab')
        for adsorbate in self.adsorbates:
            for metal in self.raw_energies[adsorbate]:
                try:
                    DeltaE =  self.raw_energies[adsorbate][metal] \
                            - self.raw_energies['slab'][metal] \
                            - self.references[adsorbate]
                except TypeError:
                    continue
                self.adsorption_energies[adsorbate][metal] = DeltaE
                        

def get_references(reference_nodes):
    """Get the references from the reference dict of nodes."""
    references = defaultdict(float)
    for adsorbate, reference_node in reference_nodes.items():
        node = load_node(reference_node)
        references[adsorbate] = node.outputs.output_parameters.get_attribute('energy')
    return references
        
if __name__ == '__main__':
    """Get the d-band center, band width, chemisorption energy from a DFT calculation."""
    GROUPNAME = 'transition_metals/scf/density_of_states'
    ADSORBATE = ['C', 'O']

    with open('references.json', 'r') as handle:
        reference_nodes = json.load(handle)
    
    references = get_references(reference_nodes)

    data = DataFromDFT(GROUPNAME, ADSORBATE, references)

    # Save the adsorption energies and pdos to json
    with open('output/adsorption_energies.json', 'w') as handle:
        json.dump(data.adsorption_energies, handle, indent=4)
    with open('output/pdos.json', 'w') as handle:
        json.dump(data.pdos, handle, indent=4)


