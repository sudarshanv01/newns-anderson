"""Store the lowest energy structures in a different group."""
from pprint import pprint
import sys
import aiida
from aiida import orm
from collections import defaultdict
import numpy as np

# --- aiida imports
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
DosWorkflow = WorkflowFactory('quantumespresso.pdos')
        
if __name__ == '__main__':
    """Determine the node with the lowest energy and store that."""

    groupname = sys.argv[1]
    STRUCTURES_GROUP_LABEL = 'PBE/SSSP_precision/gauss_smearing_0.1eV/sampled/relax/Al_reference/adsorbates'
    # Store the lowest energy node in a new group
    subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)

    # Create the query for this adsorbate
    qb = QueryBuilder()
    qb.append(Group, filters={'label':groupname}, tag='Group')
    qb.append(PwBaseWorkChain, with_group='Group', tag='calctype')


    results = defaultdict(lambda: defaultdict(list))

    for node in qb.all(flat=True):
        # Get the energy and the pk of the nodes
        if not node.is_finished_ok:
            raise ValueError('Node {} is not finished'.format(node.pk))
        adsorbate = node.get_extra('adsorbate')
        energy = node.outputs.output_parameters.get_attribute('energy')
        pk = node.pk
        results[adsorbate]['energy'].append(energy)
        results[adsorbate]['pk'].append(pk)

    # Get the lowest energy for each adsorbate
    for adsorbate in results:
        index_lowest = np.argmin(results[adsorbate]['energy'])
        pk_lowest = results[adsorbate]['pk'][index_lowest]
        print('{} has the lowest energy of {}'.format(adsorbate, results[adsorbate]['energy'][index_lowest])) 
        subgroup.add_nodes(load_node(pk_lowest))