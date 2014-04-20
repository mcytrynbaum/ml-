from numpy import * 
from percep import * 
from helper import * 

def perplot_online(): 

    data = read('train2k.databw.35')
    labels = read('train2k.label.35')
    
    m = len(labels)
    d = olp(data,labels)
    labels = labels.reshape(m)
    pred = d['pred']
    ahloss = d['ahloss']

    #compute running errors 
    cs = labels - pred
    cor = cs != 0 
    cor = cor.astype(float)
    cnt = array(cor)
    
    for i in range(0,m): 
        cnt[i] = sum(cor[0:i])

    #plot total errors as a function of time for online algorithm 
    plot(cnt)
    xlabel('time') 
    ylabel('total # errors')
    title('total # errors vs. time in online perceptron') 
    savefig('online_error.pdf')

    #plot average hinge loss as a function of time 
    plot(ahloss)
    xlabel('time') 
    ylabel('average hinge loss') 
    title('average hinge loss over time in online perceptron') 
    savefig('online_error_hinge.pdf')

    return(1)

def plot_crossval(): 
    data = read('train2k.databw.35')
    labels = read('train2k.label.35')

    d = crossval(data,labels)
    ec = d['ec']
    loss = d['loss']
    t = d['t']

    plot(t,ec)
    xlabel('# of steps of gradient ascent')
    ylabel('average crossvalidation error (0-1 LOSS)')  
    title('steps of gradient ascent vs. crossvalidation 0-1 LOSS') 
    savefig('t_error.pdf') 

    plot(t,loss)
    xlabel('# of steps of gradient ascent')
    ylabel('average crossvalidation error (HINGE LOSS)') 
    title('steps of gradient ascent vs. crossvalidation HINGE LOSS') 
    savefig('t_error_hinge.pdf') 
    return(1)

def comp_final(): 
    '''
    Note here that we use the optimal T-value T = 22
    '''
    
    T = 22
    tst = read('test200.databw.35')
    data = read('train2k.databw.35')
    labels = read('train2k.label.35')

    w = comp_w(data,labels,T)
    pred = comp_pred(tst,w) 
    print pred 

    return(1)
    






