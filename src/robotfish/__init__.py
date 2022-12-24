from .pipeline import *

def robotfish_entry(parser):
    print('robotfish_entry')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    parser.add_argument('--PROJECT_NAME', default='robotfish00', type=str, help=None)

    parser.add_argument('--n_iter', default=256, type=int, help=None)
    parser.add_argument('--fishmode', default=None, type=str, help=None)
    parser.add_argument('--average_every', default=24, type=int, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary

    if dargs['fishmode'] is None:
        run_fish(dargs)
    elif dargs['fishmode'] =='srd':
        run_fish_srd(dargs)
    else:
        print('fishmode not recognized')
    

