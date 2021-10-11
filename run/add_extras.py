
import os

def add_extras_from_structure(groupname, type_of_calc):
    """Add extras for calculations from the workchain to the structure."""
    qb = QueryBuilder()
    qb.append(Group, filters={'label':groupname}, tag='Group')
    qb.append(type_of_calc, with_group='Group', tag='calctype')

    for node in qb.all(flat=True):
        # identifier = node.inputs.pw.structure.get_extra_many(('dopant', 'support', 'dopant_index'))
        # node.set_extra_many({
        #     'dopant': identifier[0],
        #     'support': identifier[1],
        #     'dopant_index': identifier[2],
        # })
        identifier = node.inputs.pw.structure.get_extra('metal')
        node.set_extra('metal', identifier)

if __name__ == '__main__':

    PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
    # metals = ['Au', 'Ag', 'Cu', 'Rh', 'Pt']
    adsorbates = ['C', 'O']

    for ads in adsorbates:
        groupname = f'transition_metals/relaxed/{ads}'
        add_extras_from_structure(groupname, PwBaseWorkChain)