from src.utils import *


###############################################
#           FishSale Auction
###############################################

def manage_dir(dargs):
    print('pipeline.manage_dir')
    if dargs['ROOT_DIR'] is None:
        ROOT_DIR = os.getcwd()

    if dargs['PROJECT_NAME'] == 'auto':
        assert(dargs['EXPT_NUMBER'] is not None)
        dargs['PROJECT_NAME'] = f"fishSale-{dargs['stage']}-p{dargs['n_patrons']}-r{dargs['rarity']}-{dargs['EXPT_NUMBER']}"

    CKPT_DIR = os.path.join(ROOT_DIR, dargs['CKPT_FOLDER'])
    os.makedirs(CKPT_DIR, exist_ok=True)
    PROJECT_DIR = os.path.join(CKPT_DIR, dargs['PROJECT_NAME'])

    os.makedirs(PROJECT_DIR,exist_ok=True)

    PATRON_DIR_LIST = os.path.join(PROJECT_DIR,'patrons')
    os.makedirs(PATRON_DIR_LIST, exist_ok=True)
    PURCHASE_RECORD_DIR = os.path.join(PROJECT_DIR, 'purchase_record.json')

    OBSERVATORY_DIR = os.path.join(CKPT_DIR,'observatory')
    os.makedirs(OBSERVATORY_DIR, exist_ok=True)
    OBSERVATORY_REPORT_DIR = os.path.join(OBSERVATORY_DIR,'fishsaleSRD.txt')
    OBSERVATORY_PFC_REPORT_DIR = os.path.join(OBSERVATORY_DIR,'fishsaleSRD_PFC.txt')

    DIRS = {
        'ROOT_DIR': ROOT_DIR,
        'CKPT_DIR': CKPT_DIR,
        'PROJECT_DIR': PROJECT_DIR,
        'OBSERVATORY_DIR': OBSERVATORY_DIR,

        'PATRON_DIR_LIST': PATRON_DIR_LIST,
        'PURCHASE_RECORD_DIR': PURCHASE_RECORD_DIR,

        'OBSERVATORY_REPORT_DIR': OBSERVATORY_REPORT_DIR,
        'OBSERVATORY_PFC_REPORT_DIR': OBSERVATORY_PFC_REPORT_DIR,
    }


    return DIRS

def run_fishsale_eVauction(dargs):
    print('run_fishsale_eVauction()')
    from .fishSale import select_server, select_screener
    DIRS = manage_dir(dargs)
    PROJECT_DIR = DIRS['PROJECT_DIR']
    if os.path.exists(PROJECT_DIR):
        print(f"removing existing project folder {PROJECT_DIR} and start anew...")
        shutil.rmtree(PROJECT_DIR)

    initial_sell_price = base_price = dargs['base_price']

    ################### SCREENER ###################
    # In this local version of implementation,
    #   we store patrons' model in chekpoint/project_name folder
    patron_addresses = [f"{DIRS['PATRON_DIR_LIST']}/patron_{str(1000+i)[1:]}" \
        for i in range(dargs['n_patrons'])]

    sc = select_screener(dargs) 
    sc.patrons_screening(patron_addresses)

    ################### DATA ##################
    from .data import get_fish_for_sale_dataset    
    threshold_sell_price = 4.8
    input_vector=(initial_sell_price,1.,1.,1.,
        0.5,0.5,0.5,
        0.5) # see details in data.py
    dataset = get_fish_for_sale_dataset(input_vector=input_vector, n_variations=64)

    ################### SERVER ###################
    server = select_server(dataset, dargs)
    server.add_patron_addresses(patron_addresses)
    purchase_record = server.run_loop()

    purchase_record.update({'_dargs':dargs})
    with open(DIRS['PURCHASE_RECORD_DIR'],'w') as this_json:
        json.dump(purchase_record, this_json, indent=4, sort_keys=True)
    print(f"result saved to {DIRS['PURCHASE_RECORD_DIR']}")

def run_fishsale_observatory_and_garage(dargs):
    print('run_fishsale_observatory_and_garage')
    DIRS = manage_dir(dargs)

    if not 'sub-stage' in dargs:
        from .fishSaleObservatory import tinker_and_report
        tinker_and_report(DIRS['OBSERVATORY_REPORT_DIR'])
    elif dargs['sub-stage'] == 'pfc':
        from .fishSaleObservatory import tinker_and_report_PFC
        tinker_and_report_PFC(DIRS['OBSERVATORY_PFC_REPORT_DIR'])
    else:
        raise NotImplementedError()


def collate_result(dargs, result_type):
    print('collate_result_price_vs_rarity')
    if dargs['ROOT_DIR'] is None:
        ROOT_DIR = os.getcwd()
    CKPT_DIR = os.path.join(ROOT_DIR, dargs['CKPT_FOLDER'])

    DIRS = {
        'CKPT_DIR': CKPT_DIR,      
    }

    if result_type == 'price_vs_rarity':
        result_file = 'result_price_rarity.result'
        result_img = 'result_price_rarity.png'
        if dargs['screening_mode'] == 'compromised':
            result_file = 'compr-' + result_file
            result_img = 'compr-'+ result_img
        DIRS.update({
            'RESULT_PRICE_RARITY_DIR': os.path.join(CKPT_DIR, result_file),   
            'FIGURE_PRICE_RARITY_DIR': os.path.join(CKPT_DIR, result_img),              
            })
        from .results import _collate_result_price_vs_rarity
        _collate_result_price_vs_rarity(DIRS, dargs['name_prefix_cue'], dargs['noOptim_folders'], dargs['Optim_folders'] )
    
    elif result_type == 'purchase_rate':
        result_file = 'result_purchase_rate_rarity.result'
        result_img = 'result_purchase_rate_rarity.png'
        if dargs['screening_mode'] == 'compromised':
            result_file = 'compr-' + result_file
            result_img = 'compr-'+ result_img
        DIRS.update({
            'RESULT_PURCHASE_RATE_RARITY_DIR': os.path.join(CKPT_DIR, result_file),
            'FIGURE_PURCHASE_RATE_RARITY_DIR': os.path.join(CKPT_DIR, result_img), 
            })
        from .results import _collate_result_purchase_rate
        _collate_result_purchase_rate(DIRS, dargs['name_prefix_cue'], dargs['noOptim_folders'], dargs['Optim_folders'] )
    else:
        raise NotImplementedError()
###############################################
#           Other Auctions below
###############################################

# nothing yet