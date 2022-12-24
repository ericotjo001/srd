from .model_half_cheetah_design_stage import *
from .model_half_cheetah_utils import *


def half_cheetah_model_design(dargs):
    print('half_cheetah_model_design')

    if dargs['stage'] == 1:
        stage_one_design(dargs)
    elif dargs['stage'] == 2:
        stage_two_design(dargs)
    elif dargs['stage'] == 3:
        stage_three_design(dargs)
    elif dargs['stage'] == 4:
        stage_four_design(dargs)
    else:
        raise NotImplementedError()
    return

def half_cheetah_expt(dargs):
    print('half_cheetah_expt')
    DIRS = manage_srd_dirs(dargs)
    dce = DataCollectionSRD(dargs=dargs, DIRS=DIRS)
    dce.one_run()

def half_cheetah_vis(dargs):
    print('half_cheetah_vis')
    DIRS = manage_srd_visualization_dirs(dargs)

    rearranged = {
        0.: { # inhibitor
            'on': {}, # srd
            'off': {},
        },
        2.: { # inhibitor
            'on': {}, # srd
            'off': {},
        }
    }
    for FOLDER_DIR in glob.glob(DIRS['MODEL_FOLDER_DIR'] + f"/{dargs['exptcodename']}*"):
        DATA_DIRS = glob.glob(FOLDER_DIR + f"/simulation-*.data")
        for i in range(len(DATA_DIRS)):
            data = joblib.load(DATA_DIRS[i])
            # print(data.keys()) 
            # dict_keys(['exptlabel', 'bswing', 'inhibitor', 'srd', 'x_ref', 'tf'])

            inhibitor = data['inhibitor']
            srd = data['srd']
            bswing = data['bswing']
            exptlabel = data['exptlabel']

            if not bswing in rearranged[inhibitor][srd]:
                rearranged[inhibitor][srd][bswing] = []
            rearranged[inhibitor][srd][bswing].append(data['x_ref'])
    
    plt.figure(figsize=(9,8))
    font = {'size' : 9}
    matplotlib.rc('font', **font)
    count = 1
    for srd in ['on', 'off']:
        for inhibitor in [0.,2.0]:
            plt.gcf().add_subplot(2,2,count)
            plot_bswing_var(rearranged[inhibitor][srd], inhibitor, srd)
            count += 1            
    plt.tight_layout()
    plt.savefig(DIRS['VIS_IMG_DIR'])
    print(f"visualization image saved to {DIRS['VIS_IMG_DIR']}")

def plot_bswing_var(data_per_inhibitor_per_srd, inhibitor, srd):
    dat = data_per_inhibitor_per_srd
    ndat = len(dat)
    for j, (bswing, x_ref_data) in enumerate(dat.items()):
        # print(bswing, len(x_ref_data), len(x_ref_data[0])) # 5.0 2 3600
        
        x_ref_data = np.array(x_ref_data).T # (t, n_expt)
        x_mean = np.mean(x_ref_data, axis=1) 
        x_deviation = np.var(x_ref_data,axis=1)

        label = f'{bswing}' 
        col = (j/ndat,0,1-j/ndat)
        t_ = range(len(x_mean))
        plt.gca().plot(t_ , x_mean, c=col, label=label, linewidth=1)
        plt.gca().fill_between(t_, x_mean-x_deviation, x_mean+x_deviation,color=col, alpha=0.1)
        plt.gca().set_xlabel('time')
        plt.gca().set_ylabel('x position')
    plt.gca().set_title(f'inhibitor:{inhibitor} (srd {srd})')
    plt.legend(title='backswing')
    
     

####################################
#     Self Reward Design EXPT
####################################

"""  
The half cheetah model here is the final version of models tested in "model design experiments". 
Through stage 1 to 4 of the "model design experiments", we observe and test different components
and input, manually arranging the model so that the neural network model is maximally interpretable.

This is half of the core principle in Self Reward Design. After this, we make a few more finer 
adjustments, and then include self-reward mechanism. 

What is the self reward in this case? In standard Mujoco half-cheetah problem, the training is
aimed to increase the reward, which is, roughly speaking, how fast can the cheetah run. At the
beginning, the cheetah can't even run. With reinforcement learning, the cheetah slowly learn to run. In our design, with manually selected components, the cheetah already knows how to run.

So, what do we optimize? 

########### update ###########

"""

class HalfCheetahSRD(nn.Module):
    def __init__(self,INIT_PARAMS_DIR,
        interval=25,
        BACKLEG_BACKSWING=5.,
        noise_factor=0.01):
        super(HalfCheetahSRD, self).__init__()
        """
        To understand the details, visit the 4 stages of model design. 
        This model is a further refinement of those models.
        """
        if not os.path.exists(INIT_PARAMS_DIR):
            raise RuntimeError('Initial params not available. The params can be obtained by running the command for stage 1 design.')
        init_params = joblib.load(INIT_PARAMS_DIR) 
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
            ]))).to(torch.float)
        s = self.fc.weight.data.shape
        self.fc.weight.data = self.fc.weight.data + noise_factor*torch.randn(size=s)
        self.fc.bias.data = (self.fc.bias.data*0.).to(torch.float)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.BACKLEG_BACKSWING = BACKLEG_BACKSWING  
        self.inhibitor = 0.

        # lag parameters
        self.lagcount = 0
        self.interval = interval
        self.prev_ctrl = None

        ######### PFC + self reward part #########
        # Pre-frontal cortex, used to decide if the decision is correct
        # This is the latter half of SRD design: PFC decides that the
        # decision made is correct or wrong

        self.pfc_layer1 = nn.Linear(2,2,bias=False)
        self.pfc_layer1.weight.data = torch.tensor([[-1,1.],[1.,1.]]).to(torch.float)
        self.pfc_layer2 = nn.Linear(2,2,bias=True)
        self.pfc_layer2.weight.data = torch.tensor([[1.,0.01],[-0.01,1.]]).to(torch.float)
        self.pfc_layer2.bias.data = torch.tensor([0.,-0.2]).to(torch.float)
        self.tanh_pfc = nn.Tanh()

    def propagate_first_layer(self,x):
        # x is pytorch tensor of shape (b,12)
        xs = self.xS(x[:,:6].clone())
        zs = self.zS(x[:,6:].clone())

        xsinv = -xs + 1. 
        zsinv = -zs + 1.

        x = torch.stack((xs,zs, xsinv, zsinv, self.BACKLEG_BACKSWING * x[:,6]), dim=1)
        return x
        
    def forward(self, x):
        x1 = self.propagate_first_layer(x)
        # print(x1.shape) # (1,4)

        x1 = self.leakyrelu(x1-self.inhibitor)

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

    def momentum_srd(self, x):
        self.lagcount+=1

        if self.lagcount < self.interval and self.prev_ctrl is not None:
            y1 = None
            ctrl = self.prev_ctrl 
        else:
            self.lagcount = 0

            y = self.forward(x)    
            y1 = self.self_reward(y)
            
            ctrl = y[0].clone().detach().cpu().numpy() 
            self.prev_ctrl = ctrl
        return ctrl, y1

    def self_reward(self,y):
        # print(y.shape) # torch.Size([1, 6])
        Ac = torch.mean(y**2,dim=1)
        # print('Ac:',Ac)
        iN = torch.tensor(self.inhibitor).unsqueeze(0).to(device=Ac.device)
        y1 = torch.stack((iN,Ac), dim=1)
        # print('y1:',y1)
        y1 = self.pfc_layer1(y1)
        y1 = self.tanh(4*y1)
        y1 = self.pfc_layer2(y1)
        return y1


class DataCollectionSRD():
    # this is the upgraded version of DataCollectionEnvironment from src.mujoco.model_half_cheetah_utils.py 
    def __init__(self, **kwargs):
        super(DataCollectionSRD, self).__init__()

        self.dargs = kwargs['dargs'] # all the command line arguments
        self.DIRS = kwargs['DIRS'] # directories

        self.xml = get_half_cheetah_model(x_pos = -2., floor_z_pos=-0.5)
        self.control_model = HalfCheetahSRD(self.DIRS['INIT_PARAMS_DIR'],
            BACKLEG_BACKSWING=self.dargs['bswing']).to(device=device)


        dargs = self.dargs
        self.data = {
            'exptlabel': dargs['exptlabel'],
            'bswing':dargs['bswing'],
            'inhibitor': dargs['inhibitor'],
            'srd': dargs['srd'],
            
            # position data
            'x_ref': [],
            # pfc data
            'tf': [],
        }

        self.optimizer = optim.SGD(self.control_model.parameters(), lr=0.0001, )
        self.criterion = nn.CrossEntropyLoss()
        
    def one_run(self):
        model = mujoco.MjModel.from_xml_string(self.xml)
        data = mujoco.MjData(model)

        renderer = mujoco.Renderer(model)
        cam = mujoco.MjvCamera()
        cam.distance = 5.

        mujoco.mj_resetData(model, data)
        t_ = 0
        frames = []
        while data.time < self.dargs['duration']:
            mujoco.mj_step(model, data)
            if len(frames) < data.time * self.dargs['framerate']:
                renderer.update_scene(data, camera=cam)
                pixels = renderer.render().copy()
                frames.append(pixels)
            
            data.ctrl = self.get_control_signals(data.xpos, t_)
            t_ += 1        

        self.process_collected_data(frames=frames)

    def get_control_signals(self, xpos, t):
        x = format_data_to_pytorch_tensor(xpos).to(device=device).to(torch.float)
        # print(x.shape) # torch.Size([1, 12])

        self.data['x_ref'].append(xpos[1,0] + 0.) # torso x coord

        if t>100:
            ######################## inhibitor ###########################
            # Setting inhibitor>0 makes the cheetah slow down at a regular interval 
            # More specifically, with the given SRD model, setting inhibitor=2
            #   will cause the cheetah to stop altogether at those intervals.
            # Set it to zero if you won't want this effect
            ##############################################################
            self.control_model.inhibitor = 0.
            if t>500: # alternate motion. Half cheetah stops at a regular interval
                if t%1000<600: self.control_model.inhibitor = self.dargs['inhibitor']            

            if self.dargs['srd'] == 'off':
                ctrl = self.control_model.momentum(x)            
            elif self.dargs['srd'] == 'on':

                ctrl, y1 = self.control_model.momentum_srd(x) 

                if not y1 is None:
                    self.control_model.zero_grad()
                    loss = self.criterion(y1,torch.argmax(y1,dim=1))
                    loss.backward()
                    self.optimizer.step()

                    v = y1.clone()[0].detach().cpu().numpy()
                    self.data['tf'].append(v)
            else:                
                raise NotImplementedError()
        else:
            ctrl = [0.,0,0,0,0,0]
        return ctrl

    def process_collected_data(self, frames=None):
        # assert(frames is not None)
        # plt.figure()
        # plt.plot(self.activations['xs'], c='b', label='xs')
        # plt.plot(self.activations['zs'], c='g',label='zs')
        # plt.plot(self.activations['xsinv'], c='b', label='xsinv', linestyle='dotted')
        # plt.plot(self.activations['zsinv'], c='g',label='zsinv', linestyle='dotted')        
        # plt.gca().set_ylim([-0.1,1.1])
        # plt.legend()
        # plt.savefig(self.DIRS['STAGE_TWO_PLOT_DIR'])
        # print(f"activations saved to {self.DIRS['STAGE_TWO_PLOT_DIR']}")

        joblib.dump(self.data ,self.DIRS['DATA_DIR'])

        media.write_video(self.DIRS['VIDEO_MOTION_DIR'], frames, fps=self.dargs['framerate'])
        print(f"video saved to {self.DIRS['VIDEO_MOTION_DIR']}")        