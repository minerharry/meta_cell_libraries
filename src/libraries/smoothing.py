import math
import numpy as np


class _Mavg:
    def __call__(self,*args,power=None,**kwargs):
        if power is not None:
            return (self**power)(*args,**kwargs)
        return me(*args,**kwargs);

    def __pow__(self,k:int):
        return exp_avg(self,k);

def me(x, w, include_edges=True):
    if len(x) < w:
        return np.array(x);
    avgd = np.convolve(x, np.ones(w), 'valid') / w
    if include_edges and w > 1:
        #w-1 to exclude the point where the average is placed into the final array
        start = int((w-1)/2);
        end = math.ceil((w-1)/2);
        avgd = np.insert(avgd,0,x[:start]);
        avgd = np.append(avgd,x[-end:]);
    return avgd;

a = 6
b = 7
moving_average = mavg = _Mavg();

def exp_avg(c,k:int):
    def apply(x,w,**kwargs):
        for _ in range(k):
            x = c(x,w,**kwargs);
        return x;
    return apply;

