"""Get data needed for the Newns-Anderson model from the DFT calculation."""
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

def get_density_of_states_for_node(node, element, angular_momentum=2, fermi_energy=0.0, position=None):
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
                print(orb_dict['position'])
                pdos += np.array(pdos_data[index][1])
        else:
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
        # Add the slab to the adsorbate
        # if 'slab' not in self.adsorbates:
        #     self.adsorbates.append('slab')
        
        self.get_raw_energies()
        self.get_DFT_chemisorption_energy()
    
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
            metal = metal[~np.isin(metal, self.adsorbates)]

            return metal, energy, fermi_energy, ase_structure 
        
        elif type_calc == 'relax':
            energy = self.node.outputs.output_parameters.get_attribute('energy')

            # Get information about structure
            structure = self.node.outputs.output_structure
            ase_structure = structure.get_ase()
            metal = np.unique(ase_structure.get_chemical_symbols())
            metal = metal[~np.isin(metal, self.adsorbates)]

            return metal, energy, None, ase_structure 

    def get_raw_energies(self):
        """Get the raw energy from the SCF calculation."""
        for i, adsorbate in enumerate(self.adsorbates):
            print(f'Getting energies for {adsorbate}')
            groupname = self.base_group_name[i]
            if 'dos' in groupname:
                type_of_calc = DosWorkflow 
                type_calc = 'dos'
            else:
                type_of_calc = PwBaseWorkChain
                type_calc = 'relax'

            # Create the query for this adsorbate
            qb = QueryBuilder()
            qb.append(Group, filters={'label':groupname}, tag='Group')
            qb.append(type_of_calc, with_group='Group', tag='calctype')

            for node in qb.all(flat=True):

                if not node.is_finished_ok:
                    continue

                # Get the energy
                self.node = node
                metal, energy, fermi_energy, ase_structure = self.get_scf_energies(adsorbate, type_calc=type_calc)

                print(f"{adsorbate} {metal} ")
                assert len(metal) == 1
                self.raw_energies[adsorbate][metal[0]] = energy 

                if adsorbate == 'slab':
                    # get the index of the metal atom on the surface
                    # So pick an index with the highest z value
                    index = np.argmax(ase_structure.get_positions()[:,2])
                    # get the positions of that index
                    position = ase_structure.get_positions()[index]
                    # Get the sp states of the metal atom
                    energies, pdos_p = get_density_of_states_for_node(node, metal, 
                                                angular_momentum=1, 
                                                fermi_energy=fermi_energy, 
                                                position=position)
                    energies, pdos_s = get_density_of_states_for_node(node, metal, 
                                                angular_momentum=0, 
                                                fermi_energy=fermi_energy, 
                                                position=position)
                    # Store the d states, if they exist
                    try:
                        energies, pdos_d = get_density_of_states_for_node(node, metal, 
                                                    angular_momentum=2, 
                                                    fermi_energy=fermi_energy, 
                                                    position=position)
                    except IndexError:
                        pdos_d = np.zeros(len(energies))
                    # Sum up sp states
                    pdos_sp = np.array(pdos_p) + np.array(pdos_s)
                    # Store the pdos
                    pdos_to_store = [ energies, list(pdos_d), list(pdos_sp) ]
                    
                elif adsorbate != 'slab' and 'dos' in groupname:
                    # There are density of states in this node 
                    # but the calculation is for the adsorbate on the slab
                    # For the s-states
                    # energies, pdos_s = get_density_of_states_for_node(node, adsorbate,
                    #                                                   angular_momentum=0,
                    #                                                   fermi_energy=fermi_energy)
                    # For the p-states
                    energies, pdos_p = get_density_of_states_for_node(node, adsorbate,
                                                                      angular_momentum=1,
                                                                      fermi_energy=fermi_energy)
                    sum_dos = np.array(pdos_p) #+ np.array(pdos_s)
                    pdos_to_store = [ energies, list(sum_dos) ]
                elif adsorbate != 'slab' and 'dos' not in groupname:
                    # We just need the energies here, so we will just ignore
                    # the pdos option here
                    continue

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
                    print(f"Error: {adsorbate} {metal}")
                    continue
                self.adsorption_energies[adsorbate][metal] = DeltaE
                        

def get_references(reference_nodes, functional='PBE'):
    """Get the references from the reference dict of nodes."""
    references = defaultdict(float)
    for adsorbate, reference_node in reference_nodes[functional].items():
        node = load_node(reference_node)
        references[adsorbate] = node.outputs.output_parameters.get_attribute('energy')
    return references
        
if __name__ == '__main__':
    """Get the d-band center, band width, chemisorption energy from a DFT calculation."""
    GROUPNAMES = yaml.safe_load(open('root_groups.yaml'))['groups'] 

    for root_group in GROUPNAMES:

        ADSORBATES = ['slab', 'C', 'O']
        # create the groups
        groups = []
        for adsorbate in ADSORBATES:
            groups.append(f'{root_group}/{adsorbate}')

        ROOT_FUNCTIONAL = root_group.split('/')[0]
        LABEL = root_group.replace('/', '_')
        print(LABEL, ROOT_FUNCTIONAL)
        print(groups)

        # References are just the atoms in vacuum
        with open('references.json', 'r') as handle:
            reference_nodes = json.load(handle)
        references = get_references(reference_nodes, functional=ROOT_FUNCTIONAL)

        # Get the data from the DFT calculations done with AiiDA
        data = DataFromDFT(groups, ADSORBATES, references)

        # Save the adsorption energies and pdos to json
        with open(f'output/adsorption_energies_{LABEL}.json', 'w') as handle:
            json.dump(data.adsorption_energies, handle, indent=4)

        # Save the pdos
        with open(f'output/pdos_{LABEL}.json', 'w') as handle:
            json.dump(data.pdos, handle, indent=4)