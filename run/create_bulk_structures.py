"""Create bulk structures."""
import numpy as np
from ase import build
from ase.data import covalent_radii, atomic_numbers
from aiida import orm

if __name__ == '__main__':
    """Create the initial bulk structures to be stored."""

    # Structures to be stored in this group
    STRUCTURES_GROUP_LABEL = 'initial_bulk_structures/reference'

    # Metals to consider
    # metals = [ 'Sc', 'Ti', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu', 
    #            'Y',  'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 
    #            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
    #            'Al']
    metals = [ 'Al' ]

    for metal in metals:
        print('Creating initial bulk structure for {m}'.format(m=metal))
        try:
            ase_structure = build.bulk(metal, cubic=True)
        except RuntimeError:
            ase_structure = build.bulk(metal)
        
        # Write bulk structures to the folder output/initial_bulk_structures
        ase_structure.write('output/initial_bulk_structures/{m}_initial_bulk.cif'.format(m=metal))

        StructureData = DataFactory('structure')
        structure = StructureData(ase=ase_structure)
        structure.store()

        structure.set_extra('metal', metal)
        structure.set_extra('crystalstructure', 'mixed')

        subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)
        subgroup.add_nodes(structure)