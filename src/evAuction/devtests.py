from src.utils import *
def testdummypatron():
    print('testdummypatron')
    from .fishSale import DummyPatron
    patron = DummyPatron(buy_price=5.0)
    with torch.no_grad():
        for price in [4.5,4.9,5, 5.05, 5.1,5.5]:
            x = torch.tensor([price, 1.,1.,1.,0.5,0.5,0.5,0.5, 0.5]).unsqueeze(0)
            y = patron(x)
            print(f"price: %-4s, decision vector:%s"%(str(price), str(y.numpy())))

        for length in [0.5, 0.9, 1, 1.1, 1.5]:
            x = torch.tensor([5., length,1.,1.,0.5,0.5,0.5,0.5, 0.5]).unsqueeze(0)
            y = patron(x)
            print(f"length: %-4s, decision vector:%s"%(str(length), str(y.numpy())))


def testsrdpatron():
    """
    This is just for simple dev test. Make sure things are working, the shapes
    of input/output are correct etc
    """
    print('testsrdpatron')
    from .fishSale import SRDFishSaleNegotiator

    ES_config = {
        'buy_price':5,
        'implicit_contrastive_dim':5,
        'delta' : 0.001,}
    fc_config = {
        'weight': np.array([
            [-1.,1.,1.,1.],
            [0.8,-1.,0,0],
            [2.,0,0,0] 
        ]),
        'bias': np.array([1.0,0.,-0.5])
    }
    model_config = {
        'ES_config': ES_config,
        'fc_config': fc_config,
    }

    net = SRDFishSaleNegotiator(**model_config)

    with torch.no_grad():
        x = torch.tensor([5., 1.,1.,1.,0.5,0.5,0.5, 0.5]).unsqueeze(0)
        print('\nsingle (batch=1):')
        xaug = net.implicit_augmentation(x)
        y_es = net._external_sensory_response(xaug)
        y = net.fc(y_es.transpose(1,2))
        print(y_es.shape, y.shape) # torch.Size([1, 4, 5]) torch.Size([1, 5, 3])
        
        # self reward part
        y_pfc_input = torch.cat((y_es.transpose(1,2),y), dim=2) # y_pfc_input.shape: torch.Size([1, 5, 7])
        y_pfc_normed = net.pfc_softmax(y_pfc_input)
        y_sem = net.pfc_semantic_layer(y_pfc_normed)
        print('y_pfc_input.shape:',y_pfc_input.shape)
        print('y_sem.shape:', y_sem.shape) # torch.Size([1, 5, 3])

        y_srd = net.fc_pfc(net.tanh(y_sem))
        print('y_srd.shape:',y_srd.shape) # torch.Size([1, 5, 2])


        xb = torch.tensor([
            [5.2, 1.,1.,1.,0.5,0.5,0.5,0.5,],
            [5.2, 1.,1.,1.,0.5,0.5,0.5,0.5,],
            [5.1, 1.,1.,1.,0.5,0.5,0.5,0.5,],
            [5.1, 1.,1.,1.,0.5,0.5,0.5,0.5],
            [5., 1.,1.,1.,0.5,0.5,0.5,0.5, ],
            [4.9, 1.,1.,1.,0.5,0.5,0.5,0.5,],
            [4.8, 1.,1.,1.,0.5,0.5,0.5,0.5,],
            ],)
        print('\nbatched, vary price: ', xb.shape)
        xaug = net.implicit_augmentation(xb)
        y_es = net._external_sensory_response(xaug)
        y = net.fc(y_es.transpose(1,2))
        print(y_es.shape, y.shape) # torch.Size([7, 4, 5]) torch.Size([7, 5, 3])
        
        # self reward part
        y_pfc_input = torch.cat((y_es.transpose(1,2),y), dim=2) 
        y_pfc_normed = net.pfc_softmax(y_pfc_input)
        y_sem = net.pfc_semantic_layer(y_pfc_normed)
        print('y_pfc_input.shape:',y_pfc_input.shape) # torch.Size([7, 5, 7])
        print('y_sem.shape:', y_sem.shape) # torch.Size([7, 5, 3])

        y_srd = net.fc_pfc(net.tanh(y_sem))
        print('y_srd.shape:',y_srd.shape) # torch.Size([7, 5, 2])

    