import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default=None, type=str, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    

    if args.mode == 'robotfish':
        from src.robotfish import robotfish_entry
        robotfish_entry(parser)
    elif args.mode == 'evAuction':
        from src.evAuction import evAuction_entry
        evAuction_entry(parser) 
    else:
        print('please select the correct mode.')