
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

    MOVE_LIST = ['Sc', 'Ti', 'V', 'Y', 'Zr', 'Nb', 'Hf', 'Ta', 'W', 'Mo', 'Cr']
    REPORT_LIST = []

    for node in qb.all(flat=True):
        if node.is_finished_ok:
            print('Node pk: %d completed'%node.pk)
            # Get the inputs of the calculation 
            parameters = node.inputs.scf.pw.parameters.get_dict()
            structure = node.inputs.structure
            atoms = structure.get_ase()
            chemical_symbols = atoms.symbols
            print('Chemical symbols: ', chemical_symbols)
            print('Electronic convergence: ', parameters['ELECTRONS']['conv_thr'])
            print('number of bands: ', parameters['SYSTEM']['nbnd'])
            if atoms[0].symbol in MOVE_LIST:
                REPORT_LIST.append(node.pk)
            print()

    print(REPORT_LIST)
     
        
