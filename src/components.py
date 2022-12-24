import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def selective_activation(x, epsilon=1e-4): 
    return epsilon/(epsilon+x**2)

def threshold_activation(x, a=5., c=0., negative_slope=0.01):
    return torch.tanh(F.leaky_relu(a*(x-c),negative_slope=negative_slope))

def below_threshold_activation(x,a=5., c=0., negative_slope=0.01):
    return torch.tanh(-F.leaky_relu(a*(x-c), negative_slope=negative_slope)) + 1.


class ActivationModule(nn.Module):
    def __init__(self, activation_type, epsilon=1e-4, a=5.,c=0, negative_slope=0.01):
        super(ActivationModule, self).__init__()
        self.activation_type = activation_type
        self.epsilon = epsilon
        self.a = a
        self.c = c
        self.negative_slope = negative_slope

    def forward(self,x):
        if self.activation_type == 'selective_activation':
            x = selective_activation(x)
        elif self.activation_type == 'threshold_activation':
            x = threshold_activation(x, a=self.a, c=self.c, negative_slope=self.negative_slope)
        elif self.activation_type == 'below_threshold_activation':
            x = below_threshold_activation(x, a=self.a, c=self.c, negative_slope=self.negative_slope)
        else:
            raise NotImplementedError()
        return x

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    print('components!')

    gallery_dir = 'gallery'
    os.makedirs(gallery_dir,exist_ok=True)

    x = np.linspace(-5,5,101)

    plt.figure()
    linestyles = ['solid', 'dashed']
    for i,c in enumerate([0,1]):
        for a in [1,2,5,10]:    
            y1 = threshold_activation(torch.tensor(x),a=a,c=c).numpy()
            plt.plot(x,y1, label=f'a={str(a)} c={str(c)}', linestyle=linestyles[i])
    plt.gca().set_ylim([-2.,2.])
    plt.title('threshold_activation')
    plt.legend()
    plt.savefig(os.path.join(gallery_dir, 'threshold_activation.png'))


    plt.figure()
    linestyles = ['solid', 'dashed']
    for i,c in enumerate([0,1]):
        for a in [1,2,5,10]:    
            y1 = below_threshold_activation(torch.tensor(x),a=a,c=c).numpy()
            plt.plot(x,y1, label=f'a={str(a)} c={str(c)}', linestyle=linestyles[i])
    plt.gca().set_ylim([-2.,2.])
    plt.title('below_threshold_activation')
    plt.legend()
    plt.savefig(os.path.join(gallery_dir, 'below_threshold_activation.png'))