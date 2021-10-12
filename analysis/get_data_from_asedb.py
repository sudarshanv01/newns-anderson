"""Get all the data needed for Newns-Anderson from the ASE-db."""
from ase.db import connect
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
import yaml
from pprint import pprint
import json
@dataclass
class DataFromASE:
    databasename: str
    referencesname: str
    adsorbates: list

    def __post_init__(self):
        """Get all the data from the ASE-db so that it can be stored 
        in the same format as the AiiDA database to be fed into the Newns
        Anderson model."""
        self.db = connect(self.databasename)
        self.raw_energies = defaultdict(lambda: defaultdict(dict))
        self.pdos = defaultdict(lambda: defaultdict(dict))
        self.adsorption_energy = defaultdict(lambda: defaultdict(dict)) 

        self.references = yaml.safe_load(open(self.referencesname))

        self.get_raw_energies_from_database()
        self.get_pdos_from_database()
        self.get_adsorption_energy()

    def get_raw_energies_from_database(self):
        """Get the raw energies from the ASE-db."""
        for row in self.db.select():
            adsorbate = row.states.replace('state_','')
            metal = row.sampling.replace('sampling_','')
            self.raw_energies[adsorbate][metal] = row.energy
        
    def get_pdos_from_database(self):
        """Extract the pdos from only slab calculations and put them in a dictionary."""
        for row in self.db.select():
            adsorbate = row.states.replace('state_','')
            metal = row.sampling.replace('sampling_','')
            if adsorbate == 'slab':
                self.pdos[adsorbate][metal] = [ row.data.pdos['energies'], row.data.pdos['metal'] ]

    def get_adsorption_energy(self):
        """Get the adsorption energy from the ASE database."""
        for adsorbate in self.adsorbates:
            for metal in self.raw_energies[adsorbate].keys():
                self.adsorption_energy[adsorbate][metal] = self.raw_energies[adsorbate][metal] \
                                                  - self.raw_energies['slab'][metal] - self.references[adsorbate]


if __name__ == '__main__':
    """Get the adsorption energy, projected density of states from the ASE-db."""
    databasename = 'inputs/transition_metals.db'
    referencesname = 'inputs/references_asedb.yaml'
    adsorbates = ['C', 'O']

    data = DataFromASE(databasename, referencesname, adsorbates)

    # Store the adsorption energies with as json files
    with open('output/adsorption_energies_RPBE.json', 'w') as f:
        json.dump(data.adsorption_energy, f, indent=4)

    # Store the projected density of states with as json files
    with open('output/pdos_RPBE.json', 'w') as f:
        json.dump(data.pdos, f, indent=4)