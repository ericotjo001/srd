from .utils import *

def manage_dirs(dargs):
    OBSERVATORY_DIR = 'checkpoint/mujoco/observatory'
    os.makedirs(OBSERVATORY_DIR, exist_ok=True)

    DIRS = {
        'OBSERVATORY_DIR': OBSERVATORY_DIR,
    }
    return DIRS

def view_object_info(dargs):
    if dargs['model'] == 'half-cheetah':
        from .model import get_half_cheetah_model
        xml = get_half_cheetah_model(x_pos = -1.2345, floor_z_pos=-0.5)
    else:
        raise NotImplementedError()

    model = mujoco.MjModel.from_xml_string(xml)
    print(model) # <mujoco._structs.MjModel object at 0x000001D2D84395F0>

    data = mujoco.MjData(model)
    print(data) # <mujoco._structs.MjData object at 0x000001F02266C4F0>

    renderer = mujoco.Renderer(model) # <mujoco.renderer.Renderer object at 0x000001ECB315CC10>
    print(renderer)

    cam = mujoco.MjvCamera()
    print(cam)
    """
    <MjvCamera
      azimuth: 90.0
      distance: 2.0
      elevation: -45.0
      fixedcamid: -1
      lookat: array([0., 0., 0.])
      trackbodyid: -1
      type: 0
    >  
    """  

    mujoco.mj_resetData(model, data)
    mujoco.mj_step(model, data)
    print(data.qpos)
    """
    [ 0.    -0.001 -0.     0.    -0.    -0.     0.     0.    -0.   ]
    """

    print(data.xpos) # nbody x 3
    """
    [[ 0.     0.     0.   ]
     [-1.234  0.     0.7  ] # position of <body name="torso" pos="{x_pos} 0 {z_pos}">
     [-1.734  0.     0.7  ] # <body name="bthigh" pos="-.5 0 0"> # note: this is relative to torso
     [-1.575  0.     0.45 ] # <body name="bshin" pos=".16 0 -.25">
     [-1.855  0.     0.31 ] # <body name="bfoot" pos="-.28 0 -.14">
     [-0.734  0.     0.7  ] # <body name="fthigh" pos=".5 0 0"> # note: this is relative to torso
     [-0.874  0.     0.46 ] # <body name="fshin" pos="-.14 0 -.24">
     [-0.744  0.     0.28 ]]  # <body name="ffoot" pos=".13 0 -.18">
    """

def vary_control_strength_half_cheetah(dargs):
    print('vary_control_strength_half_cheetah')
    DIRS = manage_dirs(dargs)

    duration = 4  # (seconds)
    framerate = 15  # (Hz)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.77
    def get_one_instance(actuator_str=1.):
        from .model import get_half_cheetah_model
        xml = get_half_cheetah_model(x_pos = -1, floor_z_pos=-0.5)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model)
        cam = mujoco.MjvCamera()
        cam.distance = 4.

        mujoco.mj_resetData(model, data)
        t_ = 0
        posx = []
        frames = []
        while data.time < duration:
            mujoco.mj_step(model, data)
            if len(frames) < data.time * framerate:
                renderer.update_scene(data, camera=cam)
                pixels = renderer.render().copy()

                cv2.putText(pixels, f"a:{actuator_str}", (47,47), 
                    font, fontScale, (255,0,0), 2)
                frames.append(pixels)
                # print(pixels.shape) # (240, 320, 3)

            if t_>=100  and t_<150:
                if t_% 50 < 25 :
                    data.ctrl = [0.,0,0,-2*actuator_str,0,0] # this makes front thigh swing ahead
                else:
                    data.ctrl = [-actuator_str,0,0,0,0,0] # this makes hind thigh swing ahead

            posx.append(data.xpos[1,0])

            t_ += 1

        print('actuator_str:', actuator_str)
        print('  total iter:', t_)
        return {'posx':posx, 'frames': frames}


    path_ = os.path.join(DIRS['OBSERVATORY_DIR'], f'vc-half-cheetah.mp4')
    plot_path_ = os.path.join(DIRS['OBSERVATORY_DIR'], f'vc-half-cheetah.png')

    ACTUATOR_STRS = [-2,-1,-0.5,0.25,0.5, 0.75, 1,2., 3.] 
    nA = len(ACTUATOR_STRS)

    results = {}
    for actuator_str in ACTUATOR_STRS:
        result = get_one_instance(actuator_str=actuator_str)
        results[actuator_str] = result

    all_frames = [[]]
    for i,actuator_str in enumerate(ACTUATOR_STRS): 
        all_frames[-1].append(results[actuator_str]['frames'])
        if i%3 == 2 and i+1<nA:   
            all_frames.append([])
    frame_rows = []       
    for rows in all_frames:
        frame_rows.append(np.concatenate(rows, axis=2)) 
    all_frames = np.concatenate(frame_rows, axis=1)
    # Simulate and display video.
    # media.show_video(frames, fps=framerate, width=512)
    media.write_video(path_, all_frames, fps=framerate)
    print(f'video saved to {path_}')

    all_posx = []
    linestyles = ['solid','dashed','dotted']
    plt.figure()
    plt.gcf().add_subplot(1,1,1)
    for i,actuator_str in enumerate(ACTUATOR_STRS):
        y = results[actuator_str]['posx']   
        plt.gca().plot(0.5*i+np.arange(len(y)), y, label=f'a: {actuator_str}', 
            linewidth=1.2, c=(i* 1/nA,0, 1-i*1/nA), linestyle=linestyles[i%len(linestyles)])
    plt.gca().set_ylabel('posx, xpos[0]')
    plt.legend()
    plt.savefig(plot_path_)
    

