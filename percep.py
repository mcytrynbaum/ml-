from pylab import * 
from numpy import * 

def olp(data, labels): 
    '''
    we input a data matrix where each row is an image as well as a label
    matrix corresponding to each row - the output is a dict containing 
    the predictions as well as the learned parameters
    NEED to put all helper functions as well as graph functions in here at
    the end 
    '''
    m = len(data)
    n = len(data[0])
    if (len(labels) != m): 
        return(0)
    
#implement online perceptron algorithm 
    ans = zeros(m)
    w = zeros(n)
    hloss = zeros(m)
    ahloss = zeros(m)
    rv = {}
    mcount = 0

    for i in range(0,m): 
        x = data[i] 
        t = labels[i] 
        pd = dot(x,w)

        npd = 0
        if norm(w) != 0: 
            npd = pd / (norm(w) * norm(x))
            
        #compute classifier function
        if pd > 0: 
            p = 1
        else: 
            p = -1
        
        ans[i] = p

        #correct classifier
        if ((p == 1) and (t == -1)): 
            w = w - x
        elif ((p == -1) and (t == 1)): 
            w = w + x
    
        hloss[i] = max(0, -t * npd) 
        
    for i in range(0,m): 
        ahloss[i] = sum(hloss[0:i]) / (i + 1)

    rv['w'] = w
    rv['pred'] = ans
    rv['hloss'] = hloss 
    rv['ahloss'] = ahloss

    return(rv)
    
def comp_pred(data, w): 
    (m,n) = data.shape
    pred = zeros(m)

    for i in range(0,m): 
        xi = data[i]
        pd = dot(xi, w)

        if pd > 0: 
            pred[i] = 1
        else: 
            pred[i] = -1
        
    return(pred)

def comp_w(data, labels, T): 
    '''
    return w computed using data and fit parameter T  
    '''

    nu = .7
    rv = {}

    (m,n) = data.shape
    if((len(labels) != m)):   
        return(0)

    w = zeros(n)
    labels = labels.reshape(m)

    # for theta_t, calc data and make prediction vector
    for j in range(0,T):  

        pred = comp_pred(data,w) 
        incor = find(pred != labels)
        ni = len(incor)

        tdata = data[incor]
        tlab = labels[incor]
        dt = zeros(n)

        for k in range(0,ni): 
            dt += tlab[k] * tdata[k]
        
        w = w + nu * dt 

    return(w)

def crossval(data,labels): 
    '''
    find optimal T parameter using k-fold crossvalidation 
    returns a dict with T values and hinge loss and error count for each
    NEED to put in a loop to calc the best value for a given T 
    '''

    # Set crossvalidation parameters 
    tlen = 30 
    tu = 100
    tl = 1
    tv = floor(linspace(tl,tu,tlen))
    tv = tv.astype(int)

    (m,n) = data.shape
    labels = labels.reshape(m)

    fc = 10 
    ecv = zeros(fc)
    lossv = zeros(fc)
    
    ect = zeros(tlen)
    losst = zeros(tlen)
    rv = {}

    for i in range(0,tlen): 
        T = tv[i]
        print i

        for k in range(0,fc): 

            #form fold indices 
            lw = float(k) / fc
            hw = float(k + 1) / fc
            lind = int(floor(lw * m))
            hind = int(floor(hw * m))
            rg_tst = range(lind,hind)
            rg_data = [x for x in range(0,m) if x not in rg_tst] 

            #form cross-validation data slices 
            dat = data[rg_data] 
            datlab = labels[rg_data]
            tst = data[rg_tst]
            tstlab = labels[rg_tst]

            w = comp_w(dat,datlab,T)
            pred = comp_pred(tst,w)

            #compute loss statistics 
            ind = find(pred != tstlab) 
            ecv[k] = len(ind)
            tl = 0 
            for l in ind: 
                tl += max(0,-tstlab[l] * dot(tst[l],w)/(norm(w) * norm(tst[l])))
            lossv[k] = tl 
        
        losst[i] = mean(lossv)
        ect[i] = mean(ecv)
    
    rv['loss'] = losst
    rv['ec'] = ect 
    rv['t'] = tv 

    return(rv)



        


