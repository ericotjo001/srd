"""
we create this separate entry into our mujoco examples
to decouple installation dependencies from the main project
"""

import argparse
from src.mujoco import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default=None, type=str, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    
    if dargs['mode'] == 'devtests':
    	devtests_entry(parser)
    elif dargs['mode'] == 'srd-model-design':
        srd_model_design_stages(parser)
    elif dargs['mode'] == 'srd-expt':
        srd_expt(parser)
    elif dargs['mode'] == 'srd-expt-mass':
        srd_expt_mass(parser)
    elif dargs['mode'] == 'srd-visualize':
        srd_visualize(parser)
    else:
    	raise NotImplementedError()