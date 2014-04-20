from numpy import * 

def read(filename): 
    f = open(filename,'r')
    data = [x.strip().split() for x in f.readlines()]
    for i in range(0,len(data)): 
        for j in range(0,len(data[0])): 
            data[i][j] = float(data[i][j])

    data = array(data)
    return(data)



















