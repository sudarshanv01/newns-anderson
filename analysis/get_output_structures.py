"""Get the bulk structures."""
import numpy as np
import matplotlib.pyplot as plt
from aiida import orm
from ase import io
from collections import defaultdict
import json
# --- AiiDA imports
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
DosWorkflow = WorkflowFactory('quantumespresso.pdos')
StructureData = DataFactory('structure')

if __name__ == '__main__':
    """View the structures of the bulk."""

    # Group name for the adsorbate on the transition metal
    FUNCTIONAL = 'RPBE'
    GROUPNAME = f'{FUNCTIONAL}/SSSP_precision/bulk_structures'
    type_of_calc = PwRelaxWorkChain 

    # Get the nodes from the calculation
    qb = QueryBuilder()
    qb.append(Group, filters={'label':GROUPNAME}, tag='Group')
    qb.append(type_of_calc, with_group='Group', tag='calctype')

    all_structures = []

    for node in qb.all(flat=True):
        # get the output structure
        if not node.is_finished_ok:
            continue

        output_structure = node.outputs.output_structure
        ase_structure = output_structure.get_ase()
        all_structures.append(ase_structure)

    io.write(f'output/bulk_structures_{FUNCTIONAL}.xyz', all_structures)  