"""Make different facets of the same metal."""

from ase import build
from aiida import orm
import numpy as np
from ase.data import atomic_numbers, atomic_masses, covalent_radii

StructureData = DataFactory('structure')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')

def get_lowest_and_highest_z(structure):
    """Get the lowest and highest z coordinates of the structure."""
    z_coordinates = structure.get_positions()[:, 2]
    lowest_z = np.min(z_coordinates)
    highest_z = np.max(z_coordinates)
    return np.abs(highest_z - lowest_z)

def generate_repeat_structure(bulk_structure, facet_name, min_req_z=7., req_xy=9., vacuum=10.):
    """Generate a structure where surface provided
    which is repeated to the right dimensions."""

    # Ensure the layers are sufficient 
    layers = 1
    surface = build.surface(bulk_structure, facet_name , layers=layers, vacuum=10)
    while get_lowest_and_highest_z(surface) < min_req_z: 
        layers += 1
        surface = build.surface(bulk_structure, facet_name , layers=layers, vacuum=10)
    # Repeat the structure such that we have enough to 
    # say that we are at the dilute coverage limit 
    repeat_x = 1; repeat_y = 1
    while np.linalg.norm(surface.cell[0]) < req_xy :
        repeat_x += 1
        surface = build.surface(bulk_structure, facet_name, layers=layers, vacuum=10)
        surface = surface.repeat([repeat_x, 1, 1])
    while np.linalg.norm(surface.cell[1]) < req_xy :
        repeat_y += 1
        surface = build.surface(bulk_structure, facet_name, layers=layers, vacuum=10)
        surface = surface.repeat([1, repeat_y, 1])
    
    # Create the final surface
    surface = build.surface(bulk_structure, facet_name, layers=layers, vacuum=10)
    surface = surface.repeat([repeat_x, repeat_y, 1])
    print(np.linalg.norm(surface.cell[0]), np.linalg.norm(surface.cell[1]))
    surface.set_tags(None)

    return surface

if __name__ == '__main__':

    # Information about the structure
    STRUCTURES_FULL_GROUP_LABEL = 'PBE/SSSP_efficiency/bulk_structures'
    ADSORBATE  = 'O'
    MOL_INDEX = 0
    metal_list = ['Au', 'Pt']
    DRY_RUN = False 
    all_metal_list = []
    facets_list = ['111', '100', '110', '122', '133', '210', '211', '310', '311', '320']

    # Group into which surface structures will be added
    if ADSORBATE:
        STRUCTURES_GROUP_LABEL = f'PBE/SSSP_efficiency/facet_dependence/initial/{ADSORBATE}' 
    else:
        STRUCTURES_GROUP_LABEL = f'PBE/SSSP_efficiency/facet_dependence/initial/slab'
    if not DRY_RUN:
        subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)

    # Get the bulk structures
    query = orm.QueryBuilder()
    query.append(orm.Group, tag='group', filters={'label': STRUCTURES_FULL_GROUP_LABEL})
    query.append(PwRelaxWorkChain, tag='pwrelax', with_group='group')
    results = query.all(flat=True)

    for res in results:
        # Get the bulk structure from a vc-relax calculation
        bulk_structure = res.outputs.output_structure.get_ase()
        metal = bulk_structure.get_chemical_symbols()[0]

        if metal not in metal_list:
            continue
        
        for facet in facets_list:
            facet_name = [int(a) for a in list(facet)]
            surface = generate_repeat_structure(bulk_structure, facet_name)

            # create adsorbate
            if ADSORBATE:
                height = covalent_radii[atomic_numbers[metal]] + covalent_radii[atomic_numbers[list(ADSORBATE)[MOL_INDEX]]]
                adsorbate = build.molecule(ADSORBATE)
                surface_index = np.argmax(surface.positions[:, 2])
                build.add_adsorbate(surface, adsorbate, height, position=surface[surface_index].position[:2], mol_index=MOL_INDEX) 

            if DRY_RUN:
                # print(f'Dry run: {surface}')
                surface.write(f'output/facets/{ADSORBATE}_{metal}_{facet}.cif') 
                # bulk_structure.write(f'outputs/transition_metals/{metal}_bulk.cif')
            
            else:
                structure = StructureData(ase=surface)
                structure.store()

                structure.set_extra('metal', metal)
                structure.set_extra('facet', facet)

                subgroup.add_nodes(structure)

                print(f"Structures added to group '{STRUCTURES_GROUP_LABEL}'")
                print(f"Current group size: {len(subgroup.nodes)}")
