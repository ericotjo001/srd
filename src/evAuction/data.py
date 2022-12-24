from src.utils import *
from torch.utils.data import Dataset, DataLoader

############ data for fishSale ############

def get_fish_for_sale_dataset(**kwargs):
    dataset = FishForSaleDataset(**kwargs)    
    dataset.generate_sample_variations()
    # for i in range(4):
    #     print(dataset.__getitem__(i))
    return dataset

class FishForSaleDataset(Dataset):
    # one type of fish
    def __init__(self, input_vector=(5., 1.,1.,1.,  0.5,0.5,0.5,  0.5),
        cmin=-0.2,cmax=0.2, n_variations=16
        ):
        super(FishForSaleDataset, self).__init__()

        ##### samples to be initiated! #####
        # self.samples = None 

        self.input_vector = input_vector
        """ input_vector: [price, 
            length, weight, gill color, 
            sub-type1, sub-type2, sub-type3,
            fraction_buy]

        More details:
        price: main variable
        length, weight, gill color: continuous variables
            each of them scale with price, i.e. longer fish more expensive
            C_spec_cont=3 as components of the input
        sub-types: 
            discrete variables with 'one-hot', taking the values of 0.5 or -0.5
            C_spec_discrete=as components of the input              
        fraction_buy:
            fraction of people who vote to buy in the previous iteration 

        """
        self.cmin = cmin
        self.cmax = cmax
        self.n_variations = n_variations 

    def generate_sample_variations(self):
        # this is similar to the concept of imagination in our lavaland problem
        self.samples = np.array([self.input_vector for _ in range(self.n_variations)])
        cont_variations = np.random.uniform(self.cmin,self.cmax, 
            size=(self.n_variations-1,3))
        discrete_variations = np.random.choice([-0.5,0.5], size=(self.n_variations-1,3),
            p =(0.2,0.8))

        # the first sample (first row) is not randomized
        self.samples[1:,1:4] += cont_variations
        # print(self.samples[:,:4])
        self.samples[1:,4:7] = discrete_variations
        # print(self.samples[:,4:7])

    def __getitem__(self, i):
        return self.samples[i, :]

    def __len__(self):
        return self.n_variations

    def get_main_item_price(self):
        # return the dynamic main item price
        return self.samples.__getitem__(0)[0]

    def set_main_item_price(self, new_price):
        self.samples[:,0] = new_price

############ Other data put below this line ############