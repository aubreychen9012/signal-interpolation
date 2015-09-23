from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import pandas as pd
import math

WINDOW_LENGTH = 20
## window length to be defined specifically for each channel

## down sample data to a certain length, same function as in interpolator.py
## threshold, desired length to downsample
def down_sampling(data, threshold):
    idx = np.atleast_2d(np.asarray(range(len(data))))
    data = np.atleast_2d(np.asarray(data))
    data = np.vstack((idx,data)).T
    data = data.tolist()
    if not isinstance(data, list):
        raise Exception("incorrect data provided")
    if not isinstance(threshold, int) or threshold <= 2 or threshold >= len(data):
        raise Exception("threshold not well defined")
    for i in data:
        if not isinstance(i, list) or len(i) != 2:
            raise Exception("incorrect data provided")
    every = (len(data) - 2)/(threshold - 2)
    a = 0  # Initially a is the first point in the triangle
    next_a = 0
    max_area_point = (0, 0)
    sampled = [data[0]]  # Always add the first point
    for i in range(0, threshold-2):
        avg_x = 0
        avg_y = 0
        avg_range_start = int(math.floor((i+1)*every) + 1)
        avg_range_end = int(math.floor((i+2)*every) + 1)
        avg_rang_end = avg_range_end if avg_range_end < len(data) else len(data)
        avg_range_length = avg_rang_end - avg_range_start
        while avg_range_start < avg_rang_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1
        avg_x /= avg_range_length
        avg_y /= avg_range_length
        range_offs = int(math.floor((i+0)*every) + 1)
        range_to = int(math.floor((i+1)*every) + 1)
        point_ax = data[a][0]
        point_ay = data[a][1]
        max_area = -1
        while range_offs < range_to:
            area = math.fabs((point_ax - avg_x)*(data[range_offs][1] - point_ay) - (point_ax - data[range_offs][0])*(avg_y-point_ay))*0.5
            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs  # Next a is this b
            range_offs += 1
        sampled.append(max_area_point)  # Pick this point from the bucket
        a = next_a  # This a is the next a (chosen b)
    sampled.append(data[len(data)-1])  # Always add last
    return sampled

## generate rolling_window of desired length
## window, window length to roll
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def RunningMedian(seq, M):  
    seq = iter(seq)
    s = []   
    m = M // 2    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq,M)]    
    d = deque(s)    # Simple lambda function to handle even/odd window sizes    
    median = lambda : s[m] if bool(M&1) else (s[m-1]+s[m])*0.5
    s.sort()    
    medians = [median()]   
    for item in seq:
        old = d.popleft()          # pop oldest from left
        d.append(item)             # push newest in from right
        del s[bisect_left(s, old)] # locate insertion point and then remove old 
        insort(s, item)            # insert newest such that new sort is not required        
        medians.append(median())  
    return medians

def RunningMean(seq,N,M):
    d = deque(seq[0:M])
    means = [np.mean(d)]             # contains mean of first window
    for item in islice(seq,M,N):
        old = d.popleft()            # pop oldest from left
        d.append(item)               # push newest in from right
        means.append(np.mean(d))     # mean for current window
    return means

## get running global mean of an array
def RunningGMean(data):
     gmean = []
     l = len(data)
     for i in range(l):
          gm = np.mean(data[:i+1])
          gmean.append(gm)
     gmean = np.asarray(gmean)
     return gmean

def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

## remove trend, use "lowess" to get better estimated trend for different channel!!
def detrend(data,degree=10):
        detrended=[None]*degree
        for i in range(degree,len(data)-degree):
                chunk=data[i-degree:i+degree]
                chunk=sum(chunk)/len(chunk)
                detrended.append(data[i]-chunk)
        return detrended+[None]*degree

## build features for neuralnet interpolator, duplicated function as in interpolator.py
def build_feat(data, add_feat,window_length=WINDOW_LENGTH,):
     ## use other available channels
     feat_mean = np.nan_to_num(pd.rolling_mean(data,5))
     feat_mean = rolling_window(feat_mean, window_length)
     feat_std = np.nan_to_num(pd.rolling_std(data,5))
     feat_std = rolling_window(feat_std, window_length)
     feat_diff = np.append(0,np.diff(data))
     feat_diff = rolling_window(feat_diff,window_length)
     feat_grad = [np.gradient(i) for i in rolling_window(data,window_length)]
     feat_orig = rolling_window(data,window_length)
     feat_2ndgrad = [np.gradient(i,2) for i in rolling_window(data,window_length)]
     feat_gmean = [RunningGMean(i) for i in rolling_window(data,window_length)]
     feat_ii = rolling_window(add_feat, window_length)
     feat = np.hstack((feat_orig,feat_mean,feat_std,feat_diff,feat_grad,feat_2ndgrad,feat_gmean,feat_ii))
     feat = feat.reshape(-1,8,window_length)
     feat = feat.astype('float32')
     return feat

## get target value for neuralnet interpolator, duplicated function as interpolator.py
def get_target(data,WINDOW_LENGTH):
     target = labels(rolling_window(data,WINDOW_LENGTH))
     target = np.atleast_2d(np.asarray(target)).T
     target = target.astype('float32')
     return target

## predict data with rolling window method for get predicted data of a certain length(lag)
## x, the first rolling window
## net, the fitted NeuralNet 
## add_feat, the additional data from another channel to help prediction, same channel as in build_feat
## new_append_value, the first predicted value, default: 0, as no predicted data is generated before prediction
## window_length, same window length as in rolling_window
## lags, number of predicted values to generate
def predict_feat(x, net, add_feat,new_append_value=0, window_length=20,lags = 100):
     pos = 0
     res = []
     net = net
     if new_append_value==0:
          new_append_value = net.predict(x.reshape(-1,8,window_length))
          new_append_value = new_append_value
          res.append(new_append_value[0][0])
     while pos<lags:
          x_0 = np.append(x[0][1:],new_append_value)
          x_mean = x[1]
          x_mean_append = np.mean(x_0[-5:])
          x_mean = np.append(x_mean[1:], x_mean_append)
          x_std = x[2]
          x_std_append = np.std(x_0[-5:])
          x_std = np.append(x_std[1:],x_std_append)
          #x_acf = tsa.acf(x_0,nlags = 400)
          x_grad_append = np.gradient(x_0[-2:])
          x_grad = np.append(x[4][1:],x_grad_append[-1])
          x_2ndgrad_append = np.gradient(x_0[-2:],2)
          x_2ndgrad = np.append(x[5][1:],x_2ndgrad_append[-1])
          x_diff = np.append(x[3][1:],x_0[-1]-x_0[-2])
          x_gmean = RunningGMean(x_0)
          x_gmean = np.asarray(x_gmean)
          x_ii_append = add_feat[pos]
          x_ii = np.append(x[-1][1:],x_ii_append)
          x = np.hstack((x_0,x_mean,x_std,x_diff,x_grad,x_2ndgrad,x_gmean,x_ii))
          new_append_value = net.predict(x.reshape(-1,8,window_length))
          new_append_value = new_append_value.tolist()[0][0]
          if new_append_value>4:
               new_append_value = 4
          elif new_append_value<-5:
               new_append_value = -5
          res.append(new_append_value)
          x = x.reshape(8,window_length)
          pos+=1
     res = np.asarray(res).T
     return res

## change dtype to "float32", necessary for NeuralNet class
def float32(k):
    return np.cast['float32'](k)

## additional class for control NeuralNet learning rate
## original from tutorials by dnouri
class AdjustVariable(object):
     def __init__(self, name, start=0.03, stop=0.001):
          self.name = name
          self.start, self.stop = start, stop
          self.ls = None
     def __call__(self, nn, train_history):
          if self.ls is None:
               self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
               epoch = train_history[-1]['epoch']
               new_value = float32(self.ls[epoch - 1])
               getattr(nn, self.name).set_value(new_value)

## validation: get mean squared error of original and predicted data
def MSE(y1,y2):
     if len(y1)!=len(y2):
          print "Length unmatch"
     else:
          resid = y1-y2
          mse = sum(i**2 for i in resid)/len(y1)
     return mse

