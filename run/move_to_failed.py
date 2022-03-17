"""Move failed calculations into a separate folder called with the suffix /failed."""

import os
from aiida import orm

def move_failed_group(groupname, type_of_calc):
    """Add extras for calculations from the workchain to the structure."""
    qb = QueryBuilder()
    qb.append(Group, filters={'label':groupname}, tag='Group')
    qb.append(type_of_calc, with_group='Group', tag='calctype')

    subgroup, _ = orm.Group.objects.get_or_create(label=groupname)
    subgroup_fail, _ = orm.Group.objects.get_or_create(label=groupname + '/failed')

    for node in qb.all(flat=True):
        if node.is_failed or node.is_killed:
            subgroup_fail.add_nodes(node)
            subgroup.remove_nodes(node)

if __name__ == '__main__':

    PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
    PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
    DosWorkflow = WorkflowFactory('quantumespresso.pdos')

    qb = QueryBuilder()
    qb.append(Group, tag='Group', filters={'label':{'ilike':'PBE_spin/SSSP_precision/gauss_smearing_0.1eV/dos_scf/slab'}})
    # qb.append(Group, tag='Group', filters={'label':{'ilike':'PBE_spin/SSSP_precision/gauss_smearing_0.1eV/sampling/relax/%'}})


    results = {}
    for group in qb.all(flat=True):

        if 'failed' in group.label:
            continue

        if 'dos' in group.label:
            calculation = DosWorkflow
        else:
            calculation =  PwBaseWorkChain  # PwRelaxWorkChain, PwBaseWorkChain 
        
        print(group.label)

        groupname = group.label 
        move_failed_group(groupname, calculation)