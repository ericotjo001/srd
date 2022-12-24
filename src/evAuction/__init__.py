# explanatory voting auction
from .pipeline import *

def evAuction_entry(parser):
    print('evAuction_entry')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--ROOT_DIR', default=None, type=str, help=None)
    parser.add_argument('--CKPT_FOLDER', default='checkpoint', type=str, help=None)
    parser.add_argument('--PROJECT_NAME', default='evauction00', type=str, help=None)
    parser.add_argument('--EXPT_NUMBER', default=None, type=str, help=None)

    parser.add_argument('--auctionmode', default='fish', type=str, help=None)
    parser.add_argument('--max_iter', default=64, type=int, help=None)
    parser.add_argument('--n_patrons', default=64, type=int, help=None)
    parser.add_argument('--batch_size', default=16, type=int, help=None)
    parser.add_argument('--stage', default='auctionNoOptim', type=str, help=None)
    parser.add_argument('--screening_mode', default='uniform', type=str, help=None)

    parser.add_argument('--patron_model', default='dummy', type=str, help=None)
    parser.add_argument('--base_price', default=5., type=float, help=None)
    parser.add_argument('--min_buy_price', default=4.8, type=float, help=None)
    parser.add_argument('--threshold_sell_price', default=4.8, type=float, help=None)
    parser.add_argument('--rarity', default=0.4, type=float, help=None)
    parser.add_argument('--increment', default=0.05, type=float, help=None)
    parser.add_argument('--decrement', default=0.01, type=float, help=None)

    # for mass results
    parser.add_argument('--N_EXPT', default=1, type=int, help=None)

    # for auctionmode='result'
    parser.add_argument('--noOptim_folders', nargs='+', default=[]) 
    parser.add_argument('--Optim_folders', nargs='+', default=[]) 
    parser.add_argument('--name_prefix_cue', default='fishSale', type=str, help=None)
    parser.add_argument('--rarities', nargs='+', default=[0.5,])  
    

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary

    if dargs['auctionmode'] == 'fish':
        assert(dargs['patron_model']== 'dummy')
        from .pipeline import run_fishsale_eVauction
        run_fishsale_eVauction(dargs)
    elif dargs['auctionmode'] == 'fishSRD':
        assert(dargs['patron_model']== 'srd')
        if dargs['stage'] == 'observation':
            from .pipeline import run_fishsale_observatory_and_garage
            run_fishsale_observatory_and_garage(dargs)
        elif dargs['stage'] == 'auctionNoOptim':
            from .pipeline import run_fishsale_eVauction
            run_fishsale_eVauction(dargs)
        elif dargs['stage'] == 'observation_pfc':
            dargs['sub-stage'] = 'pfc'
            from .pipeline import run_fishsale_observatory_and_garage
            run_fishsale_observatory_and_garage(dargs)     
        elif dargs['stage'] == 'auctionOptim':
            from .pipeline import run_fishsale_eVauction
            run_fishsale_eVauction(dargs)       
        else:
            raise NotImplementedError()

    elif dargs['auctionmode'] == 'fishSRD_mass':
        from .pipeline import run_fishsale_eVauction

        if dargs['stage']=='auctionOptim':
            stage_name = 'Op'
        elif dargs['stage'] == 'auctionNoOptim':
            stage_name = 'NoOp'
        else:
            raise NotImplementedError()

        if dargs['screening_mode'] == 'compromised':
            stage_name = 'compr-' + stage_name

        for rarity in dargs['rarities']:
            for i in range(1,1+dargs['N_EXPT']):
                expt_no = str(1000+i)[1:]
                print(f'\n\nrunning EXPT {expt_no}')
                dargs.update({
                    'CKPT_FOLDER': f"checkpoint/{stage_name}_p{dargs['n_patrons']}_r{rarity}" ,
                    'PROJECT_NAME': 'auto',
                    'EXPT_NUMBER': expt_no,
                    'rarity': float(rarity),
                    })
                run_fishsale_eVauction(dargs)  
        
    elif dargs['auctionmode'] == 'result':
        if dargs['stage'] == 'price_vs_rarity':
            from .pipeline import collate_result
            collate_result(dargs, 'price_vs_rarity')
        elif dargs['stage'] == 'purchase_rate':
            from .pipeline import collate_result
            collate_result(dargs, 'purchase_rate')
        else:
            raise NotImplementedError()

    ############# testing modes ##############
    
    elif dargs['auctionmode'] == 'testdummypatron':
        from .devtests import testdummypatron
        testdummypatron()
    elif dargs['auctionmode'] == 'testsrdpatron':
        from .devtests import testsrdpatron
        testsrdpatron()
    else:
        print('fishmode not recognized')