from src.utils import *
from src.components import *

######################################
#             Screener               #
######################################
"""
Ideally, this is where human being performs manual checking on explainable model.
  Reject the model if it does not make sense or is potentially harmful.
  To facilitate such manual checking, Self Reward Design (SRD) should be 
  used since it is ensured to maximize transparency. 
"""

def select_screener(dargs):
    if dargs['patron_model'] == 'dummy':
        config = {
            'model_config':{
                'min_buy_price': dargs['min_buy_price'],
                'base_price' : dargs['base_price'],}            
            }
        from .fishSale import DummyScreenerSRD
        sc = DummyScreener(**config)
    elif dargs['patron_model'] == 'srd':
        from .fishSale import DummyScreenerSRD
        config = {
            'screening_mode': dargs['screening_mode'],
            'ES_config':{
                'base_price' : dargs['base_price'],
                'min_buy_price' : dargs['min_buy_price']
            },
            'stage': dargs['stage'],
        }
        sc = DummyScreenerSRD(**config)
    else:
        raise NotImplementedError()
    return sc

class Screener():
    def __init__(self, **kwargs):
        super(Screener, self).__init__()
        
    def patrons_screening(self, patron_addresses):
        for patron_address in patron_addresses:
            self._approve_or_reject_credential(patron_address)
        
    def _approve_or_reject_credential(self, patron_address):
        raise NotImplementedError('Implement a credential screening')
        

class DummyScreener(Screener):
    """ 
    This screener does nothing. We assume all models are accepted since
    we want to experiment on the performance. An actual screener should allow
    the server manager to reject models that do not make sense manually after 
    checking their models.
    """
    def __init__(self, **config):
        super(DummyScreener, self).__init__()
        self.config = config

    def _approve_or_reject_credential(self, patron_address):
        # dummy credential screening initiates models if they don't exist yet 
        # does no actual screening
   
        if not os.path.exists(patron_address):
            # patron = init_patron(self.config)
            model_conf = self.config['model_config']
            patron = DummyPatron(buy_price=np.random.uniform(model_conf['min_buy_price'],
                model_conf['base_price']))
            
            os.makedirs(patron_address, exist_ok=True)
            joblib.dump(patron , os.path.join(patron_address,'model.pt'))
        return None

class DummyScreenerSRD(Screener):
    """ 
    Similar to DummyScreener, but let's make the patrons a bit more fancy
    """
    def __init__(self, **config):
        Screener.__init__(self)
        self.config = config

    def patrons_screening(self, patron_addresses):
        screening_mode = self.config['screening_mode']
        if screening_mode == 'uniform':
            for patron_address in patron_addresses:
                self._approve_or_reject_credential(patron_address, screening_mode=screening_mode)
        elif screening_mode == 'compromised':
            halfnpatrons = int(len(patron_addresses)/2)
            for i,patron_address in enumerate(patron_addresses):
                smode = 'uniform' if i < halfnpatrons else 'compromised'
                self._approve_or_reject_credential(patron_address, screening_mode=smode)
        else:
            raise NotImplementedError()

    def _approve_or_reject_credential(self, patron_address, screening_mode='uniform'):
        # dummy credential screening initiates models if they don't exist yet 
        # does no actual screening

        ES_config = {
            'buy_price':np.random.uniform(
                self.config['ES_config']['min_buy_price'], 
                self.config['ES_config']['base_price']), 
            'implicit_contrastive_dim':5,
            'delta' : 0.001,}
        model_config = {
            'ES_config': ES_config,
            'fc_config':{
                'weight': None,'bias': None,
            }
        }     

        if screening_mode == 'uniform': # see tinker_and_report()
            # recall fc takes in PG, SZ, LSR, ST and output buy, hold (lower price) or quit
            hold_bias = np.random.uniform(-0.5,0.5)
            quit_bias = np.random.uniform(0.,0.5)
            model_config['fc_config'].update({
                'weight': np.array([
                        [-1.,1.,1.,1.],
                        [0.8,-1.,0,0],
                        [2.,0,0,0] 
                    ]),
                'bias': np.array([1.0,hold_bias,-0.5+quit_bias])
                },)                
            patron = SRDFishSaleNegotiator(**model_config)
        elif screening_mode =='compromised':
            # inject malicious agent
            patron = DummyPatron(buy_price=0.001)
        else:
            raise NotImplementedError()
        os.makedirs(patron_address, exist_ok=True)
        joblib.dump(patron , os.path.join(patron_address,'model.pt'))


        if self.config['stage'] == 'auctionOptim':
            srd_config = {
                'n_epochs': np.random.randint(0,3),
                'batch_size': np.random.randint(4,16),
                'learning_rate': np.random.uniform(1e-7,1e-4)
            }

            if screening_mode == 'compromised':
                srd_config = {
                    'n_epochs': 0,
                    'batch_size': 2,
                    'learning_rate': 0.,
                }

            with open(os.path.join(patron_address,'srd.json'), 'w') as json_file:
                json.dump(srd_config, json_file, indent=4, sort_keys=True)

        return None



######################################
#               Server               #
######################################

def select_server(dataset, dargs):
    if dargs['stage'] in ['auctionNoOptim', ]:
        from .fishSale import EVAuctionServer
        server = EVAuctionServer(dataset, 
            n_available=int(dargs['rarity']*dargs['n_patrons']),
            increment=dargs['increment'], 
            decrement=dargs['decrement'],
            threshold_sell_price=dargs['threshold_sell_price'],
            batch_size=dargs['batch_size'],
            )
    elif dargs['stage'] == 'auctionOptim':
        from .fishSale import EVAuctionServerWithSRD
        server = EVAuctionServerWithSRD(dataset, 
            n_available=int(dargs['rarity']*dargs['n_patrons']),
            increment=dargs['increment'], 
            decrement=dargs['decrement'],
            threshold_sell_price=dargs['threshold_sell_price'],
            batch_size=dargs['batch_size'],
            )
    else:
        raise NotImplementedError()
    return server

class EVAuctionServer():
    """
    The base class that runs standard auction procedure. 
    SRD optimization is not implemented.
    """
    def __init__(self, dataset, 
        max_iter=64, 
        n_available=7, 
        increment=0.5, 
        decrement=0.1, 
        threshold_sell_price=4.95,
        batch_size=16):
        super(EVAuctionServer, self).__init__()

        # patron_addresses to be set up
        self.patron_addresses = None 

        # dataset is from torch.utils.data import Dataset, DataLoader
        # it contains the dynamic main item price
        # to be fetched with get_main_item_price()
        self.dataset = dataset 

        self.batch_size = batch_size
        self.max_iter = max_iter
        self.n_initial_supply = n_available
        self.n_available = n_available
        assert(increment>=5*decrement)
        self.increment = increment
        self.decrement = decrement
        self.threshold_sell_price = threshold_sell_price

        self.prospective_buyers = []
        self.buyers = {}
        self.quitters = []
        self.quitters_queue = []

    def add_patron_addresses(self, patron_addresses):
        self.patron_addresses = [x for x in patron_addresses] # make sure it's a list

    def run_loop(self,):    
        n_available_record = []
        price_record = []

        for i in range(self.max_iter):
            # let's collect some data for observation purposes
            n_available_record.append(self.n_available)
            price_record.append(round(self.dataset.get_main_item_price(),5))
            
            # Evaluation starts here! Patrons make their choices here
            # This process is labelled Ev in the schemati
            
            votes = self._patrons_evaluate() # (n_data,3)
            # print(votes)
            
            # check the first conditions: how does no. of buyers compare to the supply
            # Denoted by orange arrows
            main_item_votes = votes[0] 
            N_buy = int(main_item_votes[0])
            if N_buy <= self.n_available:
                print(f'N_buy <= self.n_available: {N_buy} <= {self.n_available} qq={len(self.quitters_queue)}')
                self.low_demand_adjustment(votes)
            elif N_buy > self.n_available:
                # high demand, increase price! Purchase not made.
                print(f'N_buy > self.n_available: {N_buy} > {self.n_available} qq={len(self.quitters_queue)}')
                self.high_demand_adjustment(votes)
            else:
                raise Exception('unknown error')

            self.adjust_decision_statistics(votes)

            for b in self.quitters_queue:
                self.patron_addresses.remove(b)   
                self.quitters.append(b) 
            self.quitters_queue = []

            # items sold out!
            if self.n_available == 0: break

            # all patrons done
            if len(self.patron_addresses) == 0: break

            # check termination condition
            if self.dataset.get_main_item_price() < self.threshold_sell_price:
                print('terminate by stalemate!')
                break

            if i<=4:
                # SRD optimization
                self.srd_optimization()

        n_available_record.append(self.n_available)
        purchase_record = {
            'buyers':self.buyers, 
            'quitters': self.quitters,
            '_n_available_record': n_available_record,
            '_n_initial_supply': self.n_initial_supply,
            '_price_record': price_record
        }
        
        ############# printing just for fun ##############
        prices = {}
        for x,y in self.buyers.items(): 
            price = y['price']
            if not price in prices:
                prices[price] = 1
            else:
                prices[price] += 1
        print('\nfinal purchases:', prices)
        print(f'remaining items:{self.n_available}/{self.n_initial_supply}')
        print(f'n buyers:', len(self.buyers))
        print(f'n quitters:', len(self.quitters))
        
        return purchase_record


    def _patrons_evaluate(self,):
        n_data = self.dataset.__len__()
        votes = np.zeros(shape=(n_data, 3)) 
        assert(len(self.prospective_buyers)==0)

        print(f'_patrons_evaluate at price {round(self.dataset.get_main_item_price(),5)}')
        for patron_address in self.patron_addresses:
            this_votes = self._single_patron_evaluation(patron_address)                    
            votes += this_votes
        return votes

    def _single_patron_evaluation(self, patron_address):
        MODEL_DIR = os.path.join(patron_address, 'model.pt')
        patron = joblib.load(MODEL_DIR)
        patron.to(device=device)
        patron.eval()

        n_data = self.dataset.__len__()
        votes = np.zeros(shape=(n_data, 3)) 
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            all_votes = []
            for i,x in enumerate(dataloader):
                x = x.to(torch.float).to(device=device)
                y = patron(x)

                ####### greedy #######
                y_pred = torch.argmax(y, dim=1).cpu().numpy()
                all_votes.extend(y_pred)

                if i==0 and y_pred[0]==0:
                    self.prospective_buyers.append(patron_address)
                if i==0 and y_pred[0]==2:
                    self.quitters_queue.append(patron_address)

            for i,decision in enumerate(all_votes):
                votes[i, decision] += 1

        return votes

    def high_demand_adjustment(self, votes):
        # increase price
        self.threshold_sell_price = self.dataset.get_main_item_price() + self.decrement

        randomized_increment = self.increment - np.random.uniform(0.,4.) * self.decrement
        new_price = self.dataset.get_main_item_price() + randomized_increment 
        self.dataset.set_main_item_price(new_price)

        # no purchase is made
        self.prospective_buyers = [] # reset to nothing

    def low_demand_adjustment(self, votes):
        purchase_price = self.dataset.get_main_item_price()

        # decrease price
        randomized_decrement = self.decrement 
        new_price = self.dataset.get_main_item_price() - randomized_decrement
        self.dataset.set_main_item_price(new_price)

        # Update states
        for b in self.prospective_buyers:
            self.patron_addresses.remove(b)
            self.buyers[b] = {'price': round(purchase_price,5)}
            self.n_available -= 1
        
        self.prospective_buyers = [] # reset

    def adjust_decision_statistics(self, votes):
        # print(votes)
        tmp = np.sum(votes, axis=1)
        tmp[tmp==0] = 1
        buy_fraction = votes[:,0]/tmp
        # print('buy_fraction:',buy_fraction)
        # print(self.dataset.samples.shape)
        self.dataset.samples[:,-1] = buy_fraction

    def srd_optimization(self,):
        # no optimization implemented here
        # Implement specific optimization in a separate child class
        return

class EVAuctionServerWithSRD(EVAuctionServer):
    def __init__(self, dataset, 
        max_iter=64, 
        n_available=7, 
        increment=0.5, 
        decrement=0.1, 
        threshold_sell_price=4.95,
        batch_size=16):
        EVAuctionServer.__init__(self, dataset, 
            max_iter=max_iter, 
            n_available=n_available, 
            increment=increment, 
            decrement=decrement, 
            threshold_sell_price=threshold_sell_price,)

    def srd_optimization(self, ):
        for patron_address in self.patron_addresses:
            self._single_patron_srd_optimization(patron_address)        
        
    def _single_patron_srd_optimization(self, patron_address):
        MODEL_DIR = os.path.join(patron_address, 'model.pt')
        patron = joblib.load(MODEL_DIR)
        patron.to(device=device)
        patron.train()        

        SRD_CONFIG_DIR = os.path.join(patron_address, 'srd.json')
        with open(SRD_CONFIG_DIR) as f:
            srd_config = json.load(f)

        optimizer = optim.SGD(patron.parameters(), lr=srd_config['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(self.dataset, batch_size=srd_config['batch_size'], shuffle=True)
        niter = len(dataloader)
        for epoch in range(srd_config['n_epochs']):
            for i,x in enumerate(dataloader):
                patron.zero_grad()

                x = x.to(torch.float).to(device=device)
                y = patron.self_reward(x)

                loss = criterion(y, torch.argmax(y.clone().detach(),dim=1)) # self reward design 
                loss.backward()
                optimizer.step()

        joblib.dump(patron, MODEL_DIR)

######################################
#               Patron               #
######################################


"""
Input x is fed into patron
Output x from patron is a tensor of shape (B,C_out)

C = C_price + C_spec_cont + C_spec_discrete + 1 (see below)
  = 1 + 3 + 3 + 1 = 8

price          : current price
spec_cont      : specifications of the item on sale, continuous values (length, weight, gill color)
spec_discrete  : specifications of the item on sale, discrete values (sub-type1, sub-type2, sub-type3,)
out            : output channel. The fraction of people choosing to buy the product is used as the last input channel 
                    1. Buy
                    2. interested in buying, but want lower price 
                    3. Not interested (to quit the auction) 
                
""" 

class DummyPatron(nn.Module):
    def __init__(self, buy_price=None):
        super(DummyPatron, self).__init__()

        # some patron willing to buy at higher price, some only at lower price
        if buy_price is None:
            buy_price = np.random.uniform(4.5,5.) 

        self.price_offset = nn.Parameter(torch.tensor(buy_price))
        self.length_offset =  nn.Parameter(torch.tensor(1.))
    
    def forward(self,x):
        # Using pytorch convention, the input shape is (B,C)
        # B: batch size, C=3 as explained above
        # output shape: (B, C_out=3)
        b,c = x.shape
        y = torch.zeros(size=(b,3))
        y[:,1] = 0.35
        y[:,0] = 0.5 * (below_threshold_activation(x[:,0], a=5, c=self.price_offset) + \
            threshold_activation(x[:,1], a=5., c=self.length_offset))
        """ For your reference, 
        see devtests.py testdummypatron. This is the output:
        price: 4.5 , decision vector:[[0.512 0.35  0.   ]] # of course buy!
        price: 4.9 , decision vector:[[0.502 0.35  0.   ]]
        price: 5   , decision vector:[[0.5  0.35 0.  ]]
        price: 5.05, decision vector:[[0.378 0.35  0.   ]] # no buy
        price: 5.1 , decision vector:[[0.269 0.35  0.   ]]
        price: 5.5 , decision vector:[[0.007 0.35  0.   ]]
        length: 0.5 , decision vector:[[0.488 0.35  0.   ]] # still buy!
        length: 0.9 , decision vector:[[0.498 0.35  0.   ]]
        length: 1   , decision vector:[[0.5  0.35 0.  ]]
        length: 1.1 , decision vector:[[0.731 0.35  0.   ]]
        length: 1.5 , decision vector:[[0.993 0.35  0.   ]]
        """
        return y


class SRDFishSaleNegotiator(nn.Module):
    def __init__(self, **config):
        nn.Module.__init__(self)
        self.config = config

        # ES: external sensor is a convolution layer
        # nn.Conv1d(1,OUTPUT_CHANNELS_SIZE,KERNEL_SIZE, dilation=icd, bias=True)
        # KERNEL_SIZE=8 is input size, OUTPUT_CHANNELS_SIZE=4 for PG, SZ, LSR, ST neurons
        # icd: implicit contrastive dimension, for implicit augmentation
        self.ES = get_external_sensor(**config['ES_config'])
        self.ES_activations = nn.ModuleDict({
            'pg': ActivationModule('threshold_activation', a=20., c=1.),
            'sz': ActivationModule('threshold_activation', a=5., c=1., negative_slope=0.8),            
            'lsr': ActivationModule('threshold_activation', a=1., c=0.5),            
            'st': ActivationModule('selective_activation', epsilon=1e-5) 
        })
        
        # FC input channel is 4 for PG, SZ, LSR, ST neurons.
        # Output channel is 3 for buy, hold (want lower price), quit.
        self.fc = get_fc(**config['fc_config'])

        self.pfc_softmax = nn.Softmax(dim=2) # its input has shape (b, implicit_contrastive_dim, 7)
        

        pfc_semantic_layer = nn.Linear(7,3)
        pfc_semantic_layer.weight.data = 0. + \
            torch.tensor([[1.,0,0,0,0,1,0. ],
                [-1.,0,0,0,1,0,0. ],
                [0.,1,1,1,0,0,1 ],])
        pfc_semantic_layer.bias.data *= 0.
        self.pfc_semantic_layer = pfc_semantic_layer

        self.tanh = nn.Tanh()

        fc_pfc = nn.Linear(3,2)
        fc_pfc.weight.data =  0. + torch.tensor([
            [1.,1.,0],
            [0,0,1.]
            ])
        fc_pfc.bias.data = fc_pfc.bias.data*0 + torch.tensor([0.,0.05]) 
        self.fc_pfc = fc_pfc


    def get_noise_var(self, shape):
        noise_var = torch.randn(size=shape)
        noise_var[:,4:7] = 0.
        return 0.01 * noise_var

    def implicit_augmentation(self,x):
        # this is like contrastive learning
        # except the augmentation is done during forward propagation.
        ES_config = self.config['ES_config']
        
        # x shape is (B,C)
        x1 = torch.stack([x.clone() + (i>0) * self.get_noise_var(x.shape).to(device=x.device)
            for i in range(ES_config['implicit_contrastive_dim'])], dim=-1)
        return x1

    def _external_sensory_response(self,x1):
        b,c,icd = x1.shape
        x = x1.reshape(b,-1).unsqueeze(1) # (b,1,c*icd)
        x = self.ES(x) # (b,  C_out of ES=4, icd)
        x[:,0,:] = self.ES_activations['pg'](x[:,0,:].clone())  # clone to prevent in-place modification warning
        x[:,1,:] = self.ES_activations['sz'](x[:,1,:].clone()) 
        x[:,2,:] = self.ES_activations['lsr'](x[:,2,:].clone()) 
        x[:,3,:] = self.ES_activations['st'](x[:,3,:].clone()) 
        # to see the average neuron-wise response, use the following:
        # torch.mean(x,dim=2)
        return x

    def forward(self,x, mean_aug=True):
        # mean_aug=True will average the output across the implicit augmentation
        b,C = x.shape
        assert(C == 8) # otherwise the semantic will mess up

        x1 = self.implicit_augmentation(x)
        x_es = self._external_sensory_response(x1)
        x = self.fc(x_es.transpose(1,2)) # Linear[(b, icd, 4)] -> (b,icd, 3)

        if mean_aug:
            y_mean = torch.mean(x,dim=1) # (b,3)
            return y_mean
        # batchwise averaged decision can be found using the following:
        # y_pred = torch.argmax(y_mean, dim=1) # (b,)
        return x

    def self_reward(self, x, mean_aug=True):
        # mean_aug=True will average the output across the implicit augmentation
        xaug = self.implicit_augmentation(x)
        y_es = self._external_sensory_response(xaug)
        y = self.fc(y_es.clone().transpose(1,2))

        y_pfc_input = torch.cat((y_es.transpose(1,2),y), dim=2) # (b, icd, 7=4+3)
        y_pfc_normed = self.pfc_softmax(y_pfc_input)
        y_sem = self.pfc_semantic_layer(y_pfc_normed)  # (b,icd, 3)   
        y_srd = self.fc_pfc(self.tanh(y_sem)) # (b, icd, 2)
        if mean_aug:
            y_srd_mean = torch.mean(y_srd, dim=1)
            return y_srd_mean
        return y_srd


def get_external_sensor(**ES_config):
    # OUTPUT_CHANNELS. Here is the standard descriptions
    # 1. PG: price priority with generic quality considerations
    #    high price lights up this neuron 
    #    shorter, lighter fish with bad gill colour lights up this neuron
    #    Note: the relation is inverse, you're likely to buy at lower price and better quality fish
    #    Use threshold activation.
    # 2. SZ: sizes
    #    Longer and heavier fish lights up this neuron more strongly
    #    Use threshold activation
    # 3. LSR: limited supply rush
    #    This neuron responds to the fraction of previous purchase.
    #    If many people decided to purchase the item, the supply might run out soon! 
    #    This neuron lights up in response  
    #    Use threshold activation 
    # 4. ST: subtype
    #    This neuron respons to the subtype (discrete) variables
    #    The preferred subtype is [0.5,0.5,0.5]
    #    Use selective activation
    # this is remotely comparable to get_food_location_detectors in src/robotfish/model.py

    delta = ES_config['delta']
    small_seed = delta 

    OUTPUT_CHANNELS_SIZE = 4
    KERNEL_SIZE = 8 # same size as the input to this conv
    conv = nn.Conv1d(1,OUTPUT_CHANNELS_SIZE,KERNEL_SIZE, 
        dilation=ES_config['implicit_contrastive_dim'], bias=True)

    # print(conv.weight.data.shape) # torch.Size([4, 1, 8])
    # print(conv.bias.data.shape) # torch.Size([4])

    conv.bias.data = conv.bias.data*0. # when there is no relative bias between the neurons

    pg = np.zeros(shape=(KERNEL_SIZE,)) + small_seed
    pg[0] = 1./ES_config['buy_price'] 
    pg[1:4] -= 2*delta # length, weight and gill colour are inversely related to this neuron.
    # print('pg:',pg)
    conv.weight.data[0,:,:] = torch.from_numpy(pg).unsqueeze(0).unsqueeze(0)

    sz = np.zeros(shape=(KERNEL_SIZE,)) + small_seed
    sz[1:3] = 1./2 # only length and weight!
    conv.weight.data[1,:,:] = torch.from_numpy(sz).unsqueeze(0).unsqueeze(0)
    conv.bias.data[1] = 0.
    # print('sz:',sz)

    lsr = np.zeros(shape=(KERNEL_SIZE,)) + small_seed
    lsr[7] = 1.
    conv.weight.data[2,:,:] = torch.from_numpy(lsr).unsqueeze(0).unsqueeze(0)
    # print('lsr:',lsr)

    st = np.zeros(shape=(KERNEL_SIZE,)) + small_seed
    st[4:7] = 1./3
    conv.weight.data[3,:,:] = torch.from_numpy(st).unsqueeze(0).unsqueeze(0)
    conv.bias.data[3] = -0.5 
    # so that sub-type [0.5,0.5,0.5] will lead to value 0, 
    #   thus strong response to selective activation
    # print('st:', st)

    # print(conv.weight.data)
    # print(conv.bias.data)
    return conv


def get_fc(**fc_config):
    fc = nn.Linear(4,3)
    # print(fc.weight.data.shape) # torch.Size([3, 4])
    # print(fc.bias.data.shape) # torch.Size([3])

    # recall the four neurons activated by external sensor (ES_conv): PG, SZ, LSR and ST. 
    # they will pass through the following fully connected layer

    if fc_config['weight'] is None:
        weight = np.array([
            [0.,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ])
    else:
        weight = fc_config['weight']

    if fc_config['bias'] is None:
        bias = np.array([0.,0,0])
    else:
        bias = fc_config['bias'] 

    fc_weights = torch.tensor(weight).to(torch.float)
    fc_bias = torch.tensor(bias).to(torch.float)

    fc.weight.data = fc.weight.data*0. + fc_weights
    fc.bias.data = fc.bias.data*0. + fc_bias

    return fc        



