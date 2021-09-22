"""Create adsorbates from the metals lattice constants."""
import numpy as np
import json
from ase import Atoms
from ase import build
from ase import io
from ase import constraints
from ase.build import add_adsorbate
from ase.data import covalent_radii, atomic_numbers
from aiida import orm
# aiida specific
StructureData = DataFactory('structure')

if __name__ == '__main__':
    """Create adsorbates from the metals lattice constants 
    for different crystal strcutures and surface facets."""
    # Choose the adsorbate to create
    ADSORBATE = 'OH'
    # Closest atom to the surface
    MOL_INDEX = 0
    # Choose the group to put it in
    STRUCTURES_GROUP_LABEL = f'initial_structures/{ADSORBATE}'
    # Choose if it is a dry run
    DRY_RUN = False
    # Create or add to the group
    subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)

    # Get the contents of the json file with the bulk structures
    lattices = io.read('inputs/lattice_constants.json', ':')

    # Chosen facets
    facets = {'Sc':'001', 'Ti':'001', 'V':'110', 'Cr':'110', 
              'Fe':'110', 'Co':'001', 'Ni':'111', 'Cu':'111',
              'Y':'001', 'Zr':'001', 'Nb':'110', 'Mo':'110', 
              'Ru':'001', 'Rh':'111', 'Pd':'111', 'Ag':'111', 
              'Hf':'001', 'Ta':'110', 'W':'110', 'Re':'001',
              'Os':'001', 'Ir':'111', 'Pt':'111', 'Au':'111',
              'Al': '111'}

    for bulk_structure in lattices:
        # Only choose metals that have a calculated lattice constant
        metal = np.unique(bulk_structure.symbols)[0]
        print(f'Creating surface for {metal}')
        if facets[metal] == '001':
            layers = 2
            repeats = [6, 6, 1]
        elif facets[metal] == '110':
            layers = 4
            repeats = [4, 5, 1]
        elif facets[metal] == '100':
            layers = 3
            repeats = [4, 4, 1]
        elif facets[metal] == '111':
            layers = 4
            repeats = [3, 3, 1]
        else:
            layers = 3
            repeats = [2, 2, 1]

        facet_name = [int(a) for a in list(facets[metal])]
        surface = build.surface(bulk_structure, facet_name , layers=layers, vacuum=10)
        surface = surface.repeat(repeats)
        # surface.set_tags(None)

        # Add the adsorbate atom
        c_ase_structure = surface
        atom_closest = list(ADSORBATE)[MOL_INDEX]
        adsorbate = build.molecule(ADSORBATE)
        # adsorbate.rotate(180, 'y')
        surface_index = np.argmax(surface.positions[:, 2])
        height = covalent_radii[atomic_numbers[metal]] + covalent_radii[atomic_numbers[atom_closest]] 
        build.add_adsorbate(surface, adsorbate, height, position=surface[surface_index].position[:2], mol_index=MOL_INDEX) 

        if DRY_RUN:
            print(f'Dry run: {surface}')
            surface.write(f'output/transition_metals/surface_{metal}_{facets[metal]}.cif') 
            bulk_structure.write(f'output/transition_metals/bulk_{metal}_bulk.cif')
        else:
            structure = StructureData(ase=surface)
            structure.store()

            structure.set_extra('metal', metal)
            structure.set_extra('facet', facets[metal])

            subgroup.add_nodes(structure)

            print(f"Structures added to group '{STRUCTURES_GROUP_LABEL}'")
            print(f"Current group size: {len(subgroup.nodes)}")

