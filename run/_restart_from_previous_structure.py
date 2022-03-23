
# ------ regular imports
from ase.symbols import Symbols
from ase import Atoms
from aiida.tools.groups import GroupPath
import numpy as np
from aiida import orm
from aiida.engine import submit
from aiida.common import exceptions
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
    qb.append(BaseWorkflow, with_group='Group', tag='Screening')

    for node in qb.all(flat=True):
        if node.is_failed or node.is_killed:
            print('Node pk: %d failed'%node.pk)
            builder = node.get_builder_restart()

            try:
                print('Starting from old structure...')
                new_structure = node.called_descendants[0].outputs.output_structure
            except Exception:
                print('Starting from input structure...')
                new_structure = node.inputs.structure
            builder.pw['structure'] = new_structure

            num_machines = 4
            builder.pw.setdefault('metadata',{}).setdefault('options',{})['resources'] = {'num_machines': num_machines}
            builder.pw.setdefault('metadata',{}).setdefault('options',{})['max_wallclock_seconds'] = 50 * 60 * 60

            builder.pw.code = load_code('pw_6-7_intel2021@dtu_xeon40_home')

            calculation = submit(builder)

            # add new calculation to group
            path = GroupPath()
            path[group_name].get_group().add_nodes(calculation)

            # remove older calculation
            path[group_name].get_group().remove_nodes(node) 

            print('Removed %d from group %s and added %d'%(node.pk, group_name, calculation.pk))

            time.sleep(2)

        elif node.is_finished_ok:
            print('Node pk: %d completed'%node.pk)
        else:
            print('Node pk: %d still running'%node.pk)
        
     
        
