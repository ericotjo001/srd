import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def selective_activation(x, epsilon=1e-4): 
    return epsilon/(epsilon+x**2)

class StablePoseNeuron(nn.Module):
    def __init__(self, vector):
        super(StablePoseNeuron, self).__init__()
        self.param = nn.Parameter(torch.tensor(vector).to(torch.float))

    def forward(self, x):
        # x is a pytorch tensor of shape (b, c)

        x = torch.sum((x-self.param)**2,dim=1)
        x = selective_activation(x, epsilon=1e-4)
        return x


if __name__ == '__main__':
    print('testing stable pose neuron')
    vector = np.array([0.2,0.3,0.4, 0.5,0.6,0.7])
    m = StablePoseNeuron(vector)

    x = torch.from_numpy(vector).unsqueeze(0)
    y = m(x)
    print('======= x =======')
    print(x)
    print(y)

    xbatch = torch.cat([x + i*0.01*torch.randn(size=x.shape) for i in range(6)], dim=0)
    print('\n\n======= xbatch =======')
    print(xbatch)
    y = m(xbatch)
    print(y)