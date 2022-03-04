
# ------ regular imports
from ase.symbols import Symbols
from ase import Atoms
from aiida.tools.groups import GroupPath
import numpy as np
from aiida import orm
from aiida.engine import submit
# ------ specific imports
import sys
import time

# ------ Common workflows to pass to Query builder
RelaxWorkflow = WorkflowFactory('quantumespresso.pw.relax')
BaseWorkflow = WorkflowFactory('quantumespresso.pw.base')
DosWorkflow = WorkflowFactory('quantumespresso.pdos')
PwCalculation = CalculationFactory('quantumespresso.pw')

if __name__ == '__main__':
    """
    Check the status of all the calculations in a group
    """
    group_name = 'PBE/SSSP_precision/gauss_smearing_0.1eV/dos_scf/slab' 
    new_group_name = 'PBE/SSSP_precision/gauss_smearing_0.1eV/sampling/relax/slab' 

    path = GroupPath()

    qb = QueryBuilder()
    qb.append(Group, filters={'label':group_name}, tag='Group')
    qb.append(Node, with_group='Group', tag='Screening')

    for node in qb.all(flat=True):
        if node.is_finished_ok:
            scf_node = node.get_outgoing(BaseWorkflow).get_node_by_label('scf')
            path[new_group_name].get_group().add_nodes(scf_node)
