from .model_half_cheetah_utils import *

"""
This is where the design experiment begins!
We will go through the process step by step.
"""

############ STAGE ONE ############

def stage_one_design(dargs):
    # find the stable starting position!
    print('stage_one_design...')
    DIRS = manage_dirs(dargs)
    dce = DCEStageOne(dargs=dargs, DIRS=DIRS)
    dce.one_run()

class HalfCheetahSRDNetworkS1(nn.Module):
    # Half-cheetah
    def __init__(self,):
        super(HalfCheetahSRDNetworkS1, self).__init__()
        # in stage one design, the model does nothing. We don't need it yet.  
              

class DCEStageOne(DataCollectionEnvironment):
    def __init__(self, **kwargs):
        control_model = HalfCheetahSRDNetworkS1()
        super(DCEStageOne, self).__init__(control_model, **kwargs)

        self.positions = {
            'x': [], 'x_ref':[],
            'z': [], 'z_ref':[],
        }

    def get_control_signals(self, xpos, t):
        """
        At this point, the half-cheetah is initiated at a height
        It will drop onto the ground and stand in a stable position.
        
        For the purpose of design analysis, 
        we collect position coordinates from time 0 to some time t. At time t,
        the half cheetah has stabilized, i.e. it's standing still.

        We will later use these coordinates to plot a graph for visualization, and then
        use the position at time t for our model design.

        We're only concerned with x and z positions relative to the torso xpos[1,:]
        """
        # print(xpos.shape) # (8, 3), one instance in time
        self.positions['x'].append(xpos[1:,0] + 0.)
        self.positions['z'].append(xpos[1:,2] + 0.)
        self.positions['x_ref'].append(xpos[1,0] + 0.) # torso x coord
        self.positions['z_ref'].append(xpos[1,2] + 0.) # torso z coord

        return [0,0,0,0,0,0]

    def process_collected_data(self, frames=None):
        x = np.array(self.positions['x'])
        z = np.array(self.positions['z'])
        x_ref = np.array(self.positions['x_ref'])
        z_ref = np.array(self.positions['z_ref'])

        plot_coordinates(x,z, x_ref, z_ref, self.DIRS['STAGE_ONE_PLOT_DIR'])
        params = {
          # take the last coordinates (thus -1). The position has stabilized
          # we don't need the torso (since it's always 0 in the relative coords) 
          #   thus we take [-1,1:] instead of [-1,:]
          'x_rel': x[-1,1:] -x_ref[-1], 
          'z_rel': z[-1,1:] -z_ref[-1],
        }
        joblib.dump(params, self.DIRS['INIT_PARAMS_DIR'] )
        print(params)

def plot_coordinates(x,z, x_ref, z_ref, IMG_DIR):
    t, nparts = x.shape
    # print(t, nparts) # 1201 7


    legendprop = {'size': 8}
    alpha = 0.7
    linestyles = ['solid', 'dashed']
    plt.figure()

    plt.gcf().add_subplot(2,2,1)
    for i in range(nparts):
      plt.gca().plot(x[:,i] , 
        c=(i/nparts,0,1- i/nparts ), 
        label=f"{i}", alpha=alpha, 
        linestyle=linestyles[i%2], linewidth=1)
    plt.gca().set_ylabel('x coords')
    plt.legend(prop=legendprop)


    plt.gcf().add_subplot(2,2,2)
    for i in range(nparts):
      plt.gca().plot(z[:,i] , 
        c=(i/nparts,0,1- i/nparts ), 
        label=f"{i}", alpha=alpha, 
        linestyle=linestyles[i%2], linewidth=1)
    plt.gca().set_ylabel('z coords')
    plt.legend(prop=legendprop)        

    plt.gcf().add_subplot(2,2,3)
    for i in range(nparts):
      plt.gca().plot(x[:,i] -x_ref , 
        c=(i/nparts,0,1- i/nparts ), 
        label=f"{i}", alpha=alpha, 
        linestyle=linestyles[i%2], linewidth=1)
    plt.gca().set_ylabel('x coords, relative')
    plt.legend(prop=legendprop)


    plt.gcf().add_subplot(2,2,4)
    for i in range(nparts):
      plt.gca().plot(z[:,i] -z_ref, 
        c=(i/nparts,0,1- i/nparts ), 
        label=f"{i}", alpha=alpha, 
        linestyle=linestyles[i%2], linewidth=1)
    plt.gca().set_ylabel('z coords, relative')
    plt.legend(prop=legendprop)        

    plt.tight_layout()
    plt.savefig(IMG_DIR)


############ STAGE TWO ############

def stage_two_design(dargs):
    print('stage_two_design...')
    # test the dynamics of xS and zS neurons 
    DIRS = manage_dirs(dargs)
    dce = DCEStageTwo(dargs=dargs, DIRS=DIRS)
    dce.one_run()

class DCEStageTwo(DataCollectionEnvironment):
    def __init__(self, **kwargs):
        control_model = HalfCheetahSRDNetworkS2(kwargs['DIRS']['INIT_PARAMS_DIR']).to(device=device)
        super(DCEStageTwo, self).__init__(control_model, **kwargs)
        self.activations = {
            'xs': [], 'xsinv': [],
            'zs': [], 'zsinv': [],
        }

    def get_control_signals(self, xpos, t):
        # print(xpos.shape) # (8,3)

        x = format_data_to_pytorch_tensor(xpos).to(device=device)
        # print(x.shape) # torch.Size([1, 12])

        x1 = self.control_model.propagate_first_layer(x)

        xs = x1[0,0].item()
        zs = x1[0,1].item()
        xsinv = x1[0,2].item()
        zsinv = x1[0,3].item()
        self.activations['xs'].append(xs)
        self.activations['zs'].append(zs)
        self.activations['xsinv'].append(xsinv)
        self.activations['zsinv'].append(zsinv)

        if t>250:
            return [0,0,0,-1.,0,0]
        else:
            return [0,0,0,0.,0,0]

    def process_collected_data(self, frames=None):
        assert(frames is not None)
        plt.figure()
        plt.plot(self.activations['xs'], c='b', label='xs')
        plt.plot(self.activations['zs'], c='g',label='zs')
        plt.plot(self.activations['xsinv'], c='b', label='xsinv', linestyle='dotted')
        plt.plot(self.activations['zsinv'], c='g',label='zsinv', linestyle='dotted')        
        plt.gca().set_ylim([-0.1,1.1])
        plt.legend()
        plt.savefig(self.DIRS['STAGE_TWO_PLOT_DIR'])
        print(f"activations saved to {self.DIRS['STAGE_TWO_PLOT_DIR']}")

        media.write_video(self.DIRS['STAGE_TWO_VIDEO_DIR'], frames, fps=self.framerate)
        print(f"video saved to {self.DIRS['STAGE_TWO_VIDEO_DIR']}")

class HalfCheetahSRDNetworkS2(nn.Module):
    # Half-cheetah
    def __init__(self,INIT_PARAMS_DIR):
        super(HalfCheetahSRDNetworkS2, self).__init__()
        init_params = joblib.load(INIT_PARAMS_DIR)
        # print(init_params['x_rel'].shape) # (6,)
        # print(init_params['z_rel'].shape) # (6,)
        self.xS = StablePoseNeuron(init_params['x_rel'])
        self.zS = StablePoseNeuron(init_params['z_rel'])

    def propagate_first_layer(self,x):
        # x is pytorch tensor of shape (b,12)
        xs = self.xS(x[:,:6].clone())
        zs = self.zS(x[:,6:].clone())

        xsinv = -xs + 1. 
        zsinv = -zs + 1.

        x = torch.stack((xs,zs, xsinv, zsinv), dim=1)
        return x
        
    def forward(self, x):
        raise NotImplementedError('not yet, implement in a later stage')
        # x shape is a torch tensor of shape (b,nx+nz) 
        #   where nx and nz are the x and z coordinates of half cheetah body parts
        #   respectively. nx = nz = 7 = 3 + 3 + 1 because there are three parts making up the front legs,
        #   three parts bag legs and one part for the torso.
        return


############ STAGE THREE ############

def stage_three_design(dargs):
    print('stage_three_design...')
    DIRS = manage_dirs(dargs)
    dce = DCEStageThree(dargs=dargs, DIRS=DIRS)
    dce.one_run()

class DCEStageThree(DataCollectionEnvironment):
    def __init__(self, **kwargs):
        control_model = HalfCheetahSRDNetworkS3(kwargs['DIRS']['INIT_PARAMS_DIR']).to(device=device)
        super(DCEStageThree, self).__init__(control_model, **kwargs)

        self.positions = {
            'x': [], 'x_ref':[],
            'z': [], 'z_ref':[],
        }


    def get_control_signals(self, xpos, t):
        x = format_data_to_pytorch_tensor(xpos).to(device=device).to(torch.float)
        # print(x.shape) # torch.Size([1, 12])

        self.positions['x'].append(xpos[1:,0] + 0.)
        self.positions['z'].append(xpos[1:,2] + 0.)
        self.positions['x_ref'].append(xpos[1,0] + 0.) # torso x coord
        self.positions['z_ref'].append(xpos[1,2] + 0.) # torso z coord

        if t>100:
            ctrl = self.control_model.momentum(x)

            # y = self.control_model(x)    
            # ctrl = y[0].clone().detach().cpu().numpy()    

        else:
            ctrl = [0.,0,0,0,0,0]
        return ctrl

    def process_collected_data(self, frames=None):
        x = np.array(self.positions['x'])
        z = np.array(self.positions['z'])
        x_ref = np.array(self.positions['x_ref'])
        z_ref = np.array(self.positions['z_ref'])

        plot_coordinates(x,z, x_ref, z_ref, self.DIRS['STAGE_THREE_PLOT_DIR'])
        media.write_video(self.DIRS['STAGE_THREE_VIDEO_DIR'], frames, fps=self.framerate)
        print(f"video saved to {self.DIRS['STAGE_THREE_VIDEO_DIR']}")

class HalfCheetahSRDNetworkS3(nn.Module):
    # Half-cheetah.
    # We continue the construction of our model from HalfCheetahSRDNetworkS2
    def __init__(self,INIT_PARAMS_DIR):
        super(HalfCheetahSRDNetworkS3, self).__init__()
        init_params = joblib.load(INIT_PARAMS_DIR)
        # print(init_params['x_rel'].shape) # (6,)
        # print(init_params['z_rel'].shape) # (6,)
        self.xS = StablePoseNeuron(init_params['x_rel'])
        self.zS = StablePoseNeuron(init_params['z_rel'])

        self.fc = nn.Linear(4,6, bias=True)

        delta = 0.0001
        self.fc.weight.data = (self.fc.weight.data * 0. + \
                    torch.from_numpy(np.array([
                        [0.,0.,-0.2,-0.2], # see below **
                        [0.,0.,0.,0.],
                        [0.,0.,0,0],
                        [-1,-1,0.,0.],
                        [0.,0.,0.,0.],
                        [0.,0.,0.,0.],
                    ])) + delta).to(torch.float)
        # ** # we reduce from 1 to 0.2, because large swing of back legs cause the cheetah to settle in a stable equilibrium
        self.fc.bias.data = (self.fc.bias.data*0.).to(torch.float)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.lagcount = 0
        self.interval = 25
        self.prev_ctrl = None

    def propagate_first_layer(self,x):
        # x is pytorch tensor of shape (b,12)
        xs = self.xS(x[:,:6].clone())
        zs = self.zS(x[:,6:].clone())

        xsinv = -xs + 1. 
        zsinv = -zs + 1.

        x = torch.stack((xs,zs, xsinv, zsinv), dim=1)
        return x
        
    def forward(self, x):
        # x shape is a torch tensor of shape (b,nx+nz) 
        #   where nx and nz are the x and z coordinates of half cheetah body parts
        #   respectively. nx = nz = 6 = 3 + 3  because there are three parts making up the front legs,
        #   three parts bag legs. The coords are all measured relative to the torso.
        
        x1 = self.propagate_first_layer(x)
        # print(x1.shape) # (1,4)

        y = self.fc(x1)
        y = self.tanh(2*y)
        return y

    def momentum(self, x):
        self.lagcount+=1

        if self.lagcount < self.interval and self.prev_ctrl is not None:
            ctrl = self.prev_ctrl 
        else:
            self.lagcount = 0
            y = self.forward(x)    
            ctrl = y[0].clone().detach().cpu().numpy() 
            self.prev_ctrl = ctrl
        return ctrl

############ STAGE FOUR ############

def stage_four_design(dargs):
    print('stage_four_design...')
    DIRS = manage_dirs(dargs)
    dce = DCEStageFour(dargs=dargs, DIRS=DIRS)
    dce.one_run()

class DCEStageFour(DataCollectionEnvironment):
    def __init__(self, **kwargs):
        control_model = HalfCheetahSRDNetworkS4(kwargs['DIRS']['INIT_PARAMS_DIR']).to(device=device)
        super(DCEStageFour, self).__init__(control_model, **kwargs)

        self.positions = {
            'x': [], 'x_ref':[],
            'z': [], 'z_ref':[],
        }


    def get_control_signals(self, xpos, t):
        x = format_data_to_pytorch_tensor(xpos).to(device=device).to(torch.float)
        # print(x.shape) # torch.Size([1, 12])

        self.positions['x'].append(xpos[1:,0] + 0.)
        self.positions['z'].append(xpos[1:,2] + 0.)
        self.positions['x_ref'].append(xpos[1,0] + 0.) # torso x coord
        self.positions['z_ref'].append(xpos[1,2] + 0.) # torso z coord

        if t>100:
            ctrl = self.control_model.momentum(x)

            # y = self.control_model(x)    
            # ctrl = y[0].clone().detach().cpu().numpy()    

        else:
            ctrl = [0.,0,0,0,0,0]
        return ctrl

    def process_collected_data(self, frames=None):
        x = np.array(self.positions['x'])
        z = np.array(self.positions['z'])
        x_ref = np.array(self.positions['x_ref'])
        z_ref = np.array(self.positions['z_ref'])

        plot_coordinates(x,z, x_ref, z_ref, self.DIRS['STAGE_FOUR_PLOT_DIR'])
        media.write_video(self.DIRS['STAGE_FOUR_VIDEO_DIR'], frames, fps=self.framerate)
        print(f"video saved to {self.DIRS['STAGE_FOUR_VIDEO_DIR']}")

class HalfCheetahSRDNetworkS4(nn.Module):
    # Half-cheetah.
    # We continue the construction of our model from HalfCheetahSRDNetworkS3
    def __init__(self,INIT_PARAMS_DIR,
        interval=25,
        BACKLEG_BACKSWING=5.,
        delta=0.0001):
        super(HalfCheetahSRDNetworkS4, self).__init__()
        init_params = joblib.load(INIT_PARAMS_DIR)
        # print(init_params['x_rel'].shape) # (6,)
        # print(init_params['z_rel'].shape) # (6,)
        self.xS = StablePoseNeuron(init_params['x_rel'])
        self.zS = StablePoseNeuron(init_params['z_rel'])

        self.fc = nn.Linear(5,6, bias=True)
        self.fc.weight.data = (self.fc.weight.data * 0. + \
                    torch.from_numpy(np.array([
                        [0.,0.,-0.2,-0.2, 1], 
                        [0.,0.,0.,0., 0],
                        [0.,0.,0,0, 0],
                        [-1,-1,0.,0., 0],
                        [0.,0.,0.,0., 0],
                        [0.,0.,0.,0., 0],
                    ])) + delta).to(torch.float)
        self.fc.bias.data = (self.fc.bias.data*0.).to(torch.float)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.BACKLEG_BACKSWING = BACKLEG_BACKSWING  

        # lag parameters
        self.lagcount = 0
        self.interval = interval
        self.prev_ctrl = None


    def propagate_first_layer(self,x):
        # x is pytorch tensor of shape (b,12)
        xs = self.xS(x[:,:6].clone())
        zs = self.zS(x[:,6:].clone())

        xsinv = -xs + 1. 
        zsinv = -zs + 1.
        
        x = torch.stack((xs,zs, xsinv, zsinv, self.BACKLEG_BACKSWING * x[:,6]), dim=1)
        return x
        
    def forward(self, x):
        # x shape is a torch tensor of shape (b,nx+nz) 
        #   where nx and nz are the x and z coordinates of half cheetah body parts
        #   respectively. nx = nz = 6 = 3 + 3  because there are three parts making up the front legs,
        #   three parts bag legs. The coords are all measured relative to the torso.
        
        x1 = self.propagate_first_layer(x)
        # print(x1.shape) # (1,4)

        y = self.fc(x1)
        y = self.tanh(2*y)
        return y

    def momentum(self, x):
        self.lagcount+=1

        if self.lagcount < self.interval and self.prev_ctrl is not None:
            ctrl = self.prev_ctrl 
        else:
            self.lagcount = 0

            y = self.forward(x)    
            ctrl = y[0].clone().detach().cpu().numpy() 
            self.prev_ctrl = ctrl
        return ctrl