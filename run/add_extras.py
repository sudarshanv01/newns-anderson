
import os

def add_extras_from_structure(groupname, type_of_calc):
    """Add extras for calculations from the workchain to the structure."""
    qb = QueryBuilder()
    qb.append(Group, filters={'label':groupname}, tag='Group')
    qb.append(type_of_calc, with_group='Group', tag='calctype')

    for node in qb.all(flat=True):
        identifier = node.inputs.pw.structure.get_extra_many(('metal', 'facet'))
        print(identifier)
        node.set_extra('metal', identifier[0])
        node.set_extra('facet', identifier[1])

if __name__ == '__main__':

    PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
    adsorbates = ['C', 'O']

    for ads in adsorbates:
        groupname = f'PBE/SSSP_efficiency/relax/{ads}'
        add_extras_from_structure(groupname, PwBaseWorkChain)