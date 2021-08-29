
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
    group_name = sys.argv[1] 


    qb = QueryBuilder()
    qb.append(Group, filters={'label':group_name}, tag='Group')
    qb.append(Node, with_group='Group', tag='Screening')

    for node in qb.all(flat=True):
        if node.is_failed or node.is_killed:
            print('Node pk: %d failed'%node.pk)
        elif node.is_finished_ok:
            print('Node pk: %d completed'%node.pk)
        else:
            print('Node pk: %d still running'%node.pk)
     
        
