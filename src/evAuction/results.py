from src.utils import *

def _collate_result_price_vs_rarity(DIRS, name_prefix_cue, noOptim_folders, Optim_folders ):
    rc = ResultCollectorPR()
    ncue = len(name_prefix_cue)
    for noOp in noOptim_folders:
        noOptimExptDIR = os.path.join(DIRS['CKPT_DIR'], noOp) 
        
        for x in os.listdir(noOptimExptDIR):
            if x[:ncue] == name_prefix_cue:
                record_dir = os.path.join(noOptimExptDIR, x,'purchase_record.json')
                rc.collect_price_by_rarity_one_expt(record_dir, 'noOptim')
    rc.compute_mean_price_by_rarity(optim_option='noOptim')

    for Op in Optim_folders:
        noOptimExptDIR = os.path.join(DIRS['CKPT_DIR'], Op) 
        for x in os.listdir(noOptimExptDIR):
            if x[:ncue] == name_prefix_cue:
                record_dir = os.path.join(noOptimExptDIR, x,'purchase_record.json')
                rc.collect_price_by_rarity_one_expt(record_dir, 'Optim')
    rc.compute_mean_price_by_rarity(optim_option='Optim')

    joblib.dump(rc,DIRS['RESULT_PRICE_RARITY_DIR'])
    save_figure_price_by_rarity(DIRS['RESULT_PRICE_RARITY_DIR'], DIRS['FIGURE_PRICE_RARITY_DIR']) 

def save_figure_price_by_rarity(RESULT_PRICE_RARITY_DIR, FIGURE_PRICE_RARITY_DIR):
    rc = joblib.load(RESULT_PRICE_RARITY_DIR)

    Op = np.array(rc.price_by_rarity['Optim']) 
    Op[:,0] = Op[:,0] + np.random.uniform(-0.02,0.02, size=Op[:,0].shape)
    noOp = np.array(rc.price_by_rarity['noOptim'])
    noOp[:,0] = noOp[:,0] + np.random.uniform(-0.02,0.02, size=noOp[:,0].shape)

    plt.figure()
    plt.scatter(Op[:,0], Op[:,1], 3, alpha=0.5, facecolors='none', edgecolors=(1,0.77,0.77))
    plt.scatter(noOp[:,0], noOp[:,1], 3, alpha=0.5, facecolors='none', edgecolors=(0.77,0.77,1))

    xOptim, yOptim = rc.get_mean_price_by_rarity_array(optim_option='Optim')
    plt.plot(xOptim, yOptim, marker='x', c=(1.,0,0), label='Optim')
    xNoOptim, yNoOptim = rc.get_mean_price_by_rarity_array(optim_option='noOptim')
    plt.plot(xNoOptim, yNoOptim, marker='x', c=(0,0,1.), label='noOptim')    

    plt.gca().set_xlim(0.,1.)
    # plt.gca().set_ylim(4.8,5.4)
    plt.gca().set_xlabel('item supply')
    plt.gca().set_ylabel('purchase price')
    plt.legend()
    plt.savefig(FIGURE_PRICE_RARITY_DIR)

class ResultCollectorPR():
    def __init__(self, ):
        super(ResultCollectorPR, self).__init__()
        self.price_by_rarity = {
            'Optim': [],
            'noOptim': []
        }

        self.mean_price_by_rarity = {
            'Optim': {},
            'noOptim': {}
        }
   
    def collect_price_by_rarity_one_expt(self, record_dir, optim_option='Optim'): 
        with open(record_dir) as f:
            result_ = json.load(f)
            rarity = result_['_dargs']['rarity']
            for buyer_id, _info in result_['buyers'].items():
                self.price_by_rarity[optim_option].append([rarity, _info['price']])

                if not rarity in self.mean_price_by_rarity[optim_option]:
                    self.mean_price_by_rarity[optim_option][rarity] = [_info['price']]
                else:
                    self.mean_price_by_rarity[optim_option][rarity].append(_info['price'])

    def compute_mean_price_by_rarity(self, optim_option='Optim'):
        for rarity in self.mean_price_by_rarity[optim_option]:
            self.mean_price_by_rarity[optim_option][rarity] = np.mean(self.mean_price_by_rarity[optim_option][rarity])

    def get_mean_price_by_rarity_array(self, optim_option='Optim'):
        x,y = [], []
        for rarity, mean_price in self.mean_price_by_rarity[optim_option].items():
            x.append(rarity)
            y.append(mean_price)
        return np.array(x), np.array(y)


def _collate_result_purchase_rate(DIRS, name_prefix_cue, noOptim_folders, Optim_folders):
    print('collate_result_purchase_rate()')
    rc = ResultCollectorPurR()
    ncue = len(name_prefix_cue)
    for noOp in noOptim_folders:
        noOptimExptDIR = os.path.join(DIRS['CKPT_DIR'], noOp) 
        
        for x in os.listdir(noOptimExptDIR):
            if x[:ncue] == name_prefix_cue:
                record_dir = os.path.join(noOptimExptDIR, x,'purchase_record.json')
                rc.collect_purchase_rate_by_rarity_one_expt(record_dir, 'noOptim')
    rc.compute_mean_purchase_rate_by_rarity(optim_option='noOptim')

    for Op in Optim_folders:
        noOptimExptDIR = os.path.join(DIRS['CKPT_DIR'], Op) 
        for x in os.listdir(noOptimExptDIR):
            if x[:ncue] == name_prefix_cue:
                record_dir = os.path.join(noOptimExptDIR, x,'purchase_record.json')
                rc.collect_purchase_rate_by_rarity_one_expt(record_dir, 'Optim')
    rc.compute_mean_purchase_rate_by_rarity(optim_option='Optim')

    joblib.dump(rc,DIRS['RESULT_PURCHASE_RATE_RARITY_DIR'])
    save_figure_purchase_rate_by_rarity(DIRS['RESULT_PURCHASE_RATE_RARITY_DIR'], DIRS['FIGURE_PURCHASE_RATE_RARITY_DIR']) 

class ResultCollectorPurR(object):
    def __init__(self,):
        super(ResultCollectorPurR, self).__init__()
        self.purchase_rate_by_rarity = {
            'Optim': [],
            'noOptim': []        
        }
        self.mean_purchase_rate_by_rarity = {            
            'Optim': {},
            'noOptim': {}                
        }

    def collect_purchase_rate_by_rarity_one_expt(self, record_dir, optim_option='Optim'):
        with open(record_dir) as f:
            result_ = json.load(f)
            rarity = result_['_dargs']['rarity']
            purchase_rate = len(result_['buyers'])/result_['_dargs']['n_patrons']
            self.purchase_rate_by_rarity [optim_option].append([rarity, purchase_rate])

            if not rarity in self.mean_purchase_rate_by_rarity[optim_option]:
                self.mean_purchase_rate_by_rarity[optim_option][rarity] = [purchase_rate]
            else:
                self.mean_purchase_rate_by_rarity[optim_option][rarity].append(purchase_rate)

    def compute_mean_purchase_rate_by_rarity(self, optim_option='Optim'):
        for rarity in self.mean_purchase_rate_by_rarity[optim_option]:
            self.mean_purchase_rate_by_rarity[optim_option][rarity] = np.mean(self.mean_purchase_rate_by_rarity[optim_option][rarity])

    def get_mean_purchase_rate_by_rarity_array(self, optim_option='Optim'):
        x,y = [], []
        for rarity, mean_purchase_rate in self.mean_purchase_rate_by_rarity[optim_option].items():
            x.append(rarity)
            y.append(mean_purchase_rate)
        return np.array(x), np.array(y)

def save_figure_purchase_rate_by_rarity(RESULT_PURCHASE_RATE_RARITY_DIR, FIGURE_PURCHASE_RATE_RARITY_DIR):
    rc = joblib.load(RESULT_PURCHASE_RATE_RARITY_DIR)


    Op = np.array(rc.purchase_rate_by_rarity['Optim']) 
    opnoise = Op[:,0] + np.random.uniform(-0.02,0.02, size=Op[:,0].shape)
    noOp = np.array(rc.purchase_rate_by_rarity['noOptim'])
    noopnoise = noOp[:,0] + np.random.uniform(-0.02,0.02, size=noOp[:,0].shape)

    plt.figure()
    plt.scatter(opnoise, Op[:,1], 3, alpha=0.3, facecolors='none', edgecolors='r')
    plt.scatter(noopnoise, noOp[:,1], 3, alpha=0.3, facecolors='none', edgecolors='b')

    xOptim, yOptim = rc.get_mean_purchase_rate_by_rarity_array(optim_option='Optim')
    plt.plot(xOptim, yOptim, marker='x', c='r', label='Optim')
    xNoOptim, yNoOptim = rc.get_mean_purchase_rate_by_rarity_array(optim_option='noOptim')
    plt.plot(xNoOptim, yNoOptim, marker='x', c='b', label='noOptim')

    plt.gca().set_xlim(0.,1.)
    # plt.gca().set_ylim(4.8,5.4)
    plt.gca().set_xlabel('item supply')
    plt.gca().set_ylabel('purchase rate')
    plt.legend()
    plt.savefig(FIGURE_PURCHASE_RATE_RARITY_DIR)