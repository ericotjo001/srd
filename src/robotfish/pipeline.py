from src.utils import *

def manage_dir(dargs):
    if dargs['ROOT_DIR'] is None:
        ROOT_DIR = os.getcwd()

    CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
    os.makedirs(CKPT_DIR, exist_ok=True)
    PROJECT_DIR = os.path.join(CKPT_DIR, dargs['PROJECT_NAME'])
    os.makedirs(PROJECT_DIR,exist_ok=True)
    

    DIRS = {
        'ROOT_DIR': ROOT_DIR,
        'CKPT_DIR': CKPT_DIR,
        'PROJECT_DIR': PROJECT_DIR,
    }
    return DIRS

def run_fish(dargs):
    print('run_fish()')

    DIRS = manage_dir(dargs)

    # ========== Specify Environment ==========
    # See main text for details.
    # In short: 
    # 1. environment is a vector x = [x1,x2,x3]
    # 2. food is xk=0.5. No food is xk=0.01

    from .data_collector import FishUnitDataCollector
    dc = FishUnitDataCollector(dargs)

    from .model import Fish1D, Fish1DMapManager
    mm = Fish1DMapManager(dargs)
    fish = Fish1D(dargs, ENV_SHAPE=mm.ENV_SHAPE)

    ENV = mm.get_env_from_template()
    for i in range(dargs['n_iter']):
        x = fish.get_input_tensor(ENV=ENV)

        with torch.no_grad():
            greedy_decision = fish.make_decision(x) # returns y, x1, greedy_decision

        fish.update_state(action=greedy_decision, ENV=ENV)
        ENV = mm.update_state(action=greedy_decision, ENV=ENV)
        dc.get_unit_data(i,fish, ENV)

        if fish.INTERNAL_STATE['energy']<=0:
            print('fish is dead.')
            break

    save_dir = os.path.join(DIRS['PROJECT_DIR'],'robotfish.png')
    dc.display_data(save_dir=save_dir)   



def run_fish_srd(dargs):
    print('run_fish_srd()')
    # See run_fish(), similar but with srd
    
    assert(dargs['n_iter']>=256)
    DIRS = manage_dir(dargs)

    from .data_collector import FishUnitDataCollector
    dc = FishUnitDataCollector(dargs)

    from .model import Fish1D, Fish1DMapManager
    mm = Fish1DMapManager(dargs)
    fish = Fish1D(dargs, ENV_SHAPE=mm.ENV_SHAPE)

    MEMORY_SIZE = 8
    optimizer = optim.Adam(fish.nn.parameters(), lr=0.002, betas=(0.5,0.999))
    criterion = nn.CrossEntropyLoss()

    ENV = mm.get_env_from_template()
    z = None
    fish.nn.zero_grad()

    torch.set_printoptions(precision=2, sci_mode=False)
    for i in range(dargs['n_iter']):
        if (i+1)%128==0 or (i+1)==dargs['n_iter']:
            text = '%s/%s'%(str(i+1),str(dargs['n_iter']))
            print('%-64s'%(str(text)), end='\r')
        x = fish.get_input_tensor(ENV=ENV)

        greedy_decision, z2 = fish.make_self_rewarded_decision(x)
        z = z + z2 if z is not None else z2
        if (i+1)%MEMORY_SIZE==0:
            loss = criterion(z, torch.argmax(z,dim=1))
            loss.backward()     
            optimizer.step()   

            # reset
            z = None    
            fish.nn.zero_grad()

        fish.update_state(action=greedy_decision, ENV=ENV)
        ENV = mm.update_state(action=greedy_decision, ENV=ENV)
        
        dc.get_unit_data(i,fish, ENV)

        if fish.INTERNAL_STATE['energy']<=0:
            print('fish is dead.')
            break

    print('\nrun over')
    save_dir = os.path.join(DIRS['PROJECT_DIR'],'robotfishsrd.png')
    dc.display_srd_data(dargs, save_dir=save_dir)