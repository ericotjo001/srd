from .fishSale import *

##########################################
#       observatory and garage
#
# Do all the manual finetuning here.
##########################################


def tinker_and_report(OBSERVATORY_REPORT_DIR):  
    print('tinker_and_report')
    np.set_printoptions(precision=3, suppress=True, sign=' ')
    print_buffer = 1e-9

    report = open(OBSERVATORY_REPORT_DIR,'w')
    x_prototype = [5., 1.,1.,1.,0.5,0.5,0.5,0.5,]
    x_prototype_tensor = torch.tensor([x_prototype])
    report.write(f"Read tinker_and_report() in src/evAuction/fishSale.py to better understand this document.\n")
    report.write(f"prototype: {x_prototype}\n\n")
    
    report.write("################### EXPT 1 ###################\n")  
    # This is where we test the external sensory module
    # let's not bother about the FC layer for now.
    # Important: note that our activation functions are intended to normalize
    #   the values to [-1,1] range; see src/components.py. This will help a great 
    #   deal during the design of the next layer

    ES_config = {
        'buy_price':5,
        'implicit_contrastive_dim':5,
        'delta' : 0.001,}
    model_config = {
        'ES_config': ES_config,
        'fc_config':{
            'weight': None,'bias': None,
        }
    }
    net = SRDFishSaleNegotiator(**model_config)
    
    with torch.no_grad():
        xaug = net.implicit_augmentation(x_prototype_tensor)
        y_es = net._external_sensory_response(xaug)
        y = net.fc(y_es.transpose(1,2))
    report.write('y_es.shape:%s\n'%(str(y_es.shape)))
    report.write(f'y_es mean: %s\n'%(str(torch.mean(y_es, dim=2).numpy())))
    report.write(f'y.shape  : %s\n'%(str(y.shape)))
    report.write(f'y mean   : %s\n'%(str(torch.mean(y,dim=1).numpy())))
    report.write('\n')

    report.write("/********** external sensory responses **********/\n\n")

    ###### vary price, affect pg neuron ######
    price_delta = [-0.2,-0.1, -0.05, 0, 0.05, 0.1, 0.2]
    x_price_variations = torch.tensor([x_prototype for _ in range(len(price_delta))])
    for i,p in enumerate(price_delta):
        x_price_variations[i,0] += p
    report.write('x_price_variations:\n')
    report.write(str(x_price_variations.numpy())+'\n')
    report.write(f'x_price_variations.shape:{x_price_variations.shape}\n')
    with torch.no_grad():
        xaug = net.implicit_augmentation(x_price_variations)
        y_es = net._external_sensory_response(xaug)
    report.write(f'%-5s %s\n'%(str('price'),str(' _pg_      sz     lsr     st')))
    for p,y_es_ in zip(x_price_variations[:,0].numpy(),torch.mean(y_es, dim=2).numpy()):
        report.write(f'%-5s %s\n'%(
            str(round(p.item(),3)),str(y_es_+ print_buffer),))
    report.write('\n')


    ###### vary length and size, affect sz neuron ######
    delta = [-0.2,-0.1, -0.05, 0, 0.05, 0.1, 0.2]
    x_sz_variations = torch.tensor([x_prototype for _ in range(len(delta))])
    for i,d in enumerate(delta):
        x_sz_variations[i,1:3] += d
    report.write('x_sz_variations:\n')
    report.write(str(x_sz_variations.numpy())+'\n')
    report.write(f'x_sz_variations.shape:{x_sz_variations.shape}\n')
    with torch.no_grad():
        xaug = net.implicit_augmentation(x_sz_variations)
        y_es = net._external_sensory_response(xaug)
    report.write(f'%-7s %-7s %s\n'%(str('leng') ,str('weight'),str('  pg     _sz_     lsr     st')))
    for leng, weight, y_es_ in zip(
        x_sz_variations[:,1].numpy(),
        x_sz_variations[:,2].numpy(),
        torch.mean(y_es, dim=2).numpy()
        ):
        
        report.write(f'%-7s %-7s %s\n'%(
            str(round(leng.item(),3)),round(weight.item(),3),str(y_es_+ print_buffer),))
    report.write('\n')


    ###### vary buy/lower price fractions, affect lsr neuron ######
    delta = [-0.5, -0.25, 0, 0.25,0.5]
    x_frac_variations = torch.tensor([x_prototype for _ in range(len(delta))])
    
    for i,d in enumerate(delta):
        x_frac_variations[i,7] += d
    report.write('x_frac_variations:\n')
    report.write(str(x_frac_variations.numpy())+'\n')
    report.write(f'x_frac_variations.shape:{x_frac_variations.shape}\n')
    with torch.no_grad():
        xaug = net.implicit_augmentation(x_frac_variations)
        y_es = net._external_sensory_response(xaug)
    report.write(f'%-7s %s\n'%(str('buy') ,str('  pg      sz     _lsr_    st')))
    for frac_buy, y_es_ in zip(
        x_frac_variations[:,7].numpy(),
        torch.mean(y_es, dim=2).numpy()
        ):
        report.write(f'%-7s %s\n'%(
            str(round(frac_buy.item(),3)),y_es_ + print_buffer,))
    report.write('\n')

    ###### vary types, affect st neuron ######

    x_type_variations = torch.tensor([x_prototype for _ in range(4)])
    report.write('x_type_variations:\n')
    report.write(str(x_type_variations.numpy())+'\n')
    report.write(f'x_type_variations.shape:{x_type_variations.shape}\n')    
    x_type_variations[1,4:7] = torch.tensor([0.5,0.5,-0.5])
    x_type_variations[2,4:7] = torch.tensor([0.5,-0.5,0.5])
    x_type_variations[3,4:7] = torch.tensor([0.5,-0.5,-0.5])
    with torch.no_grad():
        xaug = net.implicit_augmentation(x_type_variations)
        y_es = net._external_sensory_response(xaug)
    report.write(f'%-18s %s\n'%(str('subtype') ,str('  pg      sz    lsr    _st_')))
    for subtype, y_es_ in zip(
        x_type_variations[:,4:7].numpy(),
        torch.mean(y_es, dim=2).numpy()
        ):
        report.write(f'%-18s %s\n'%(
            str(subtype),y_es_ + print_buffer,))
    report.write('\n')


    # Let's start experimenting with the FC layer here

    def note_down_decision_values(x,model_config):
        net = SRDFishSaleNegotiator(**model_config)
        with torch.no_grad():
            xaug = net.implicit_augmentation(x)
            y_es = net._external_sensory_response(xaug)
            y = net.fc(y_es.transpose(1,2))
            y_es_mean = torch.mean(y_es, dim=2).numpy()
            y_mean = torch.mean(y,dim=1).numpy()

        report.write('y_es.shape:%s\n'%(str(y_es.shape)))
        report.write('y.shape  : %s\n'%(str(y.shape)))
        for i in range(len(x)):
            report.write('\nx : %s \n  y_es mean: %s\n'%(str(x[i].numpy()),
                str(y_es_mean[i])))
            report.write('y mean   : %s\n'%(str(y_mean[i])))
        report.write('\n')

    report.write("################### EXPT 2.0 ###################\n") 
    # the strategy is to test the FC layer channel by channel (there are three output channels)
    # recall the four neurons: PG, SZ, LSR and ST
    # Set bias[0], i.e. bias towards making a purchase
    # The first parameter to test: weight[0,0] is negative. The stronger PG neuron lights up 
    #   (e.g. higher price) the lower is the decision to buy
    model_config = {
        'ES_config': ES_config,
        'fc_config':{
            'weight': np.array([
                [-1.,0.,0.,0], # to buy
                [0,0,0,0], # want lower price
                [0,0,0,0] ]), # not buy
            'bias': np.array([1.,0,0]),
        }
    }

    # here, we increment the price and see that the decision to buy decrease
    x = torch.tensor([
        x_prototype, # [5., 1.,1.,1.,0.5,0.5,0.5,0.5,]
        [5.1, 1.,1.,1.,0.5,0.5,0.5,0.5,],
        [5.2, 1.,1.,1.,0.5,0.5,0.5,0.5,],])
    note_down_decision_values(x,model_config)

    report.write("################### EXPT 2.1 ###################\n") 
    # let's increase some of the continuous specs to see how the decision change
    # i.e. let's update weight[0,1]
    # Stronger SZ response means longer/heavier fish with brighter gill colour (better quality). 
    #   SZ should increase with the decision to buy, thus positive weight[0,1]
    model_config['fc_config'].update({
        'weight': np.array([
                [-1.,1.,0.,0],
                [0,0,0,0],
                [0,0,0,0] ])
            })
    x = torch.tensor([
        [5.2, 1.05,1.,1.,    0.5,0.5,0.5,0.5,],
        [5.2, 1.05,1.05,1.,  0.5,0.5,0.5,0.5,],
        [5.2, 1.05,1.05,1.05,0.5,0.5,0.5,0.5,],
        [5.2, 1.1,1.1,1.05,0.5,0.5,0.5,0.5,],
        ])
    note_down_decision_values(x,model_config)

    report.write("################### EXPT 2.2 ###################\n") 
    # Now let's udpate the weight: weight[0,2] related to LSR
    # In this implementation, LSR responds strongly if more people have decided to make a purchase
    # In that case, supply may diminish quickly, so it is better for the user to make decision
    # to buy sooner than later. Thus the weight is positive.

    model_config['fc_config'].update({
        'weight': np.array([
                [-1.,1.,1.,0],
                [0,0,0,0],
                [0,0,0,0] ]),
            })
    x = torch.tensor([
        [5.2, 1.,1.,1.,0.5,0.5,0.5,0.6,],
        [5.2, 1.,1.,1.,0.5,0.5,0.5,0.8,],
        [5.2, 1.,1.,1.,0.5,0.5,0.5,0.9,],
        ])
    note_down_decision_values(x,model_config)

    report.write("################### EXPT 2.3 ###################\n") 
    # here, we set the item on sale to have a specific sub-type denoted by
    # discrete variables [0.5,0.5,0.5]
    # We want it to affect purchase decision: changing the subtype will 
    #   lower the purchase decision. From the ES layer, we know that
    #   the neuron st will respond strongly only when the subtype [0.5,0.5,0.5]
    #   You can see from the result of the following that st neuron will die off 
    #   if we change the subtype, e.g. to [0.5,0.5,-0.5]
    # Thus, udpate weight[0,3] to a positive value
    model_config['fc_config'].update({
        'weight': np.array([
                [-1.,1.,1.,1.],
                [0,0,0,0],
                [0,0,0,0] ]),
            })
    x = torch.tensor([
        [5.2, 1.,1.,1.,0.5,0.5,0.5,0.9,],
        [5.2, 1.,1.,1.,0.5,0.5,-0.5,0.9,],
        [5.2, 1.,1.,1.,0.5,-0.5,-0.5,0.9,],
        [5.2, 1.,1.,1.,0.5,-0.5,-0.5,0.5,],
        [5.2, 1.,1.,1.,0.5,0.5,-0.5,0.5,], # 2.3.5 not buy 
        [5.0, 1.,1.,1.,0.5,-0.5,-0.5,0.5,], # 2.3.6 buy
        [5.0, 1.,1.,1.,0.5,0.5,-0.5,0.5,],
        ])
    note_down_decision_values(x,model_config)


    report.write("################### EXPT 2.4 ###################\n") 
    # We have seen how the buy decision values change with the variables. 
    #   We can now  decide what values should the second output channel (don't buy, lower price)
    #   should take, so that we strike a balance betwwen buying and not buying
    # For example, here, we want case 2.3.6 to be a "buy" decision even though
    #   the subtype is different i.e. we compromise with reasonable price. 
    #   We should try different weight[1,:] values. Considering that 2.3.5 
    #   yields 0.387 in the "buy" channel but 2.3.6 yields 1.042, we want the
    #   "lower price" channel to be somewhere in between.


    model_config['fc_config'].update({
        'weight': np.array([
                [-1.,1.,1.,1.],
                [0.8,-1.,0,0],
                [0,0,0,0] ]),
            })
    x = torch.tensor([
        [5.2, 1.,1.,1.,0.5,0.5,0.5,0.9,],
        [5.2, 1.,1.,1.,0.5,0.5,-0.5,0.9,],
        [5.2, 1.,1.,1.,0.5,-0.5,-0.5,0.9,],
        [5.2, 1.,1.,1.,0.5,-0.5,-0.5,0.5,],
        [5.2, 1.,1.,1.,0.5,0.5,-0.5,0.5,], # same as 2.3.5, set to not buy 
        [5.0, 1.,1.,1.,0.5,-0.5,-0.5,0.5,], # same as 2.3.6, set buy
        [5.0, 1.,1.,1.,0.5,0.5,-0.5,0.5,],
        [5.0, 0.9,0.9,0.9,0.5,0.5,-0.5,0.5,],
        [5.0, 0.8,0.8,0.8,0.5,0.5,-0.5,0.5,],
        ])
    note_down_decision_values(x,model_config)


    report.write("################### EXPT 2.5 ###################\n") 
    # As before, weight[2,0] is positive. More precisely, if PG lights up more
    # strongly i.e. when price is higher, the second (quit) output channel should increase too.
    # We adjust the decision to quit strongly based on the price.
    # Note: the activation functions can be adjusted differently.
    # Note: recall PG, SZ, LSR and ST

    model_config['fc_config'].update({
        'weight': np.array([
                [-1.,1.,1.,1.],
                [0.8,-1.,0,0],
                [2.,0,0,0] 
            ]),
        'bias': np.array([1.0,0.,-0.5])
        },)
    x = torch.tensor([
        [5.0, 1.,1.,1.    ,0.5,0.5,-0.5,0.5,],
        [5.0, 0.9,1.,1.,   0.5,0.5,-0.5,0.5,],
        [5.0, 0.9,0.9,0.9, 0.5,0.5,-0.5,0.5,],
        [5.0, 0.8,1.,1.,   0.5,0.5,-0.5,0.5,],
        [5.0, 0.8,0.8,0.8, 0.5,0.5,-0.5,0.5,],
        [5.0, 0.8,0.8,0.8, 0.5,0.5, 0.5,0.5,],
        [5.1, 0.8,0.8,0.8, 0.5,0.5, 0.5,0.5,],
        [5.2, 0.8,0.8,0.8, 0.5,0.5, 0.5,0.5,],
        [5.2, 0.9,0.9,0.9, 0.5,0.5, 0.5,0.5,],
        [5.1, 1., 1., 1., 0.5,0.5, 0.5,0.5,],
        [5.2, 1., 1., 1., 0.5,0.5, 0.5,0.5,],
        [5.3, 1., 1., 1., 0.5,0.5, 0.5,0.5,],
        ])
    note_down_decision_values(x,model_config)

    print(f'report saved to {OBSERVATORY_REPORT_DIR}')
    report.close()



def tinker_and_report_PFC(OBSERVATORY_REPORT_DIR):  
    print('tinker_and_report_PFC')
    np.set_printoptions(precision=3, suppress=True, sign=' ')

    report = open(OBSERVATORY_REPORT_DIR,'w')
    x_prototype = [5., 1.,1.,1.,0.5,0.5,0.5,0.5,]
    x_prototype_tensor = torch.tensor([x_prototype])

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

    def note_down_pfc_values(x,model_config):
        net = SRDFishSaleNegotiator(**model_config)

        with torch.no_grad():
            xaug = net.implicit_augmentation(x)
            y_es = net._external_sensory_response(xaug)
            y_es_mean = torch.mean(y_es, dim=2).numpy()
            y = net.fc(y_es.transpose(1,2))
            y_mean = torch.mean(y,dim=1).numpy()

            y_pfc_input = torch.cat((y_es.transpose(1,2),y), dim=2) # (b, icd, 7=4+3)
            y_pfc_normed = net.pfc_softmax(y_pfc_input)
            y_pfc_normed_mean = torch.mean(y_pfc_normed, dim=1)
            y_sem = net.pfc_semantic_layer(y_pfc_normed)     
            y_sem_mean = torch.mean(y_sem, dim=1)   

            y_srd = net.fc_pfc(net.tanh(y_sem))
            y_srd_mean = torch.mean(y_srd, dim=1)


        report.write('y_pfc_input.shape:%s\n'%(str(y_pfc_input.shape)))
        for i in range(len(x)):
            report.write('\nx : %s \n'%(str(x[i].numpy()),))
            report.write('  y_es mean: %s y mean   : %s \n'%(
                str(y_es_mean[i]), str(y_mean[i]),))
            report.write('  y_pfc_normed mean: %s \n'%(
                str(y_pfc_normed_mean[i]), ))
            report.write('  y_sem mean: %s \n'%(str(y_sem_mean[i])))
            report.write('y_srd mean: %s \n'%(str(y_srd_mean[i])))
        report.write('\n')    

    note_down_pfc_values(x_prototype_tensor,model_config)
    report.write('\n')

    report.write("################### EXPT 1.1 ###################\n") 
    x = torch.tensor([
        [4.8, 1.,1.,1.,0.5,0.5,0.5,0.5,],
        [4.9, 1.,1.,1.,0.5,0.5,0.5,0.5,],
        [5., 1.,1.,1.,0.5,0.5,0.5,0.5,],
        [5.1, 1.,1.,1.,0.5,0.5,0.5,0.5,],
        [5.2, 1.,1.,1.,0.5,0.5,0.5,0.5,],
        ])
    note_down_pfc_values(x,model_config)

    report.write("################### EXPT 1.2 ###################\n") 
    x = torch.tensor([
        [5., 1.,1.,1.,0.5,0.5,0.5,0.5,],
        [5., 0.9,1.,1.,0.5,0.5,0.5,0.5,],
        [5., 0.9,0.9,0.9,0.5,0.5,0.5,0.5,],
        [5., 0.8,0.8,0.9,0.5,0.5,0.5,0.5,],

        [5., 0.9,0.9,0.9, 0.5, 0.5,-0.5,0.5,],
        [5., 0.9,0.9,0.9, 0.5,-0.5,-0.5,0.5,],
        [5., 0.9,0.9,0.9,-0.5,-0.5,-0.5,0.5,],
        ])
    note_down_pfc_values(x,model_config)



    print(f'report saved to {OBSERVATORY_REPORT_DIR}')
    report.close()