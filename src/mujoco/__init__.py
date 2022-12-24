

def srd_model_design_stages(parser):
    parser.add_argument('--model', default='half-cheetah', type=str, help=None)
    parser.add_argument('--stage', default=1, type=int, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary   
    if dargs['model'] == 'half-cheetah':
        from .model import half_cheetah_model_design
        half_cheetah_model_design(dargs)
    else:
        raise NotImplementedError()        

def srd_expt(parser):
    parser.add_argument('--model', default='half-cheetah', type=str, help=None)
    parser.add_argument('--duration', default=36, type=int, help=None) # (seconds)
    parser.add_argument('--framerate', default=15, type=int, help=None) # (Hz)
    parser.add_argument('--exptcodename', default='dtest', type=str, help=None) 

    parser.add_argument('--exptlabel', default='0', type=str, help=None) 
    parser.add_argument('--bswing', default=5., type=float, help=None) # backleg swing
    parser.add_argument('--inhibitor', default=0., type=float, help=None) # set to 2 for alternate motion
    parser.add_argument('--srd', default='off', type=str, help=None) 
    
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary   
    
    if dargs['model'] == 'half-cheetah':
        from .model import half_cheetah_expt
        half_cheetah_expt(dargs)
    else:
        raise NotImplementedError()     

def srd_expt_mass(parser):
    parser.add_argument('--model', default='half-cheetah', type=str, help=None)
    parser.add_argument('--duration', default=36, type=int, help=None) # (seconds)
    parser.add_argument('--framerate', default=15, type=int, help=None) # (Hz)
    parser.add_argument('--exptcodename', default='mass_expt_test', type=str, help=None)     

    parser.add_argument('--nexpt', default=2, type=int, help=None) 
    parser.add_argument('--bswing_grid', nargs='+', default=[3., 5.], type=float, help=None) # backleg swing
    parser.add_argument('--inhibitor_grid', nargs='+', default=[0., 2.], help=None) # set to 2 for alternate motion
    parser.add_argument('--srd_grid', nargs='+', default=['off', 'on'], help=None) 

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary   
    
    if dargs['model'] == 'half-cheetah':
        from .model import half_cheetah_expt
        for i in range(1,1+dargs['nexpt']):
            idargs = dargs.copy()
            for bswing in dargs['bswing_grid']:
                for srd in dargs['srd_grid']:
                    for inhibitor in dargs['inhibitor_grid']:
                        idargs.update({
                            'exptlabel': str(i),
                            'bswing': bswing,
                            'inhibitor': inhibitor,
                            'srd': srd, 
                        })
                        half_cheetah_expt(idargs)
    else:
        raise NotImplementedError()   

def srd_visualize(parser):
    parser.add_argument('--model', default='half-cheetah', type=str, help=None)
    parser.add_argument('--exptcodename', default='dtest', type=str, help=None) 

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary      

    if dargs['model'] == 'half-cheetah':
        from .model import half_cheetah_vis
        half_cheetah_vis(dargs)
    else:
        raise NotImplementedError()   


def devtests_entry(parser):
    print('devtests_entry')

    parser.add_argument('--testtype', default=None, type=str, help=None)
    parser.add_argument('--model', default='half-cheetah', type=str, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary   


    if dargs['testtype'] == 'object_info':
        from .observatory import view_object_info
        view_object_info(dargs)
    elif dargs['testtype'] == 'vary_control_strength':
        if dargs['model'] == 'half-cheetah':
            from .observatory import vary_control_strength_half_cheetah
            vary_control_strength_half_cheetah(dargs)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()