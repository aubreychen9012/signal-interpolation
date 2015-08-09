import preproc
import itertools
import numpy as np
import pandas as pd

## training functions

## without unit
def get_training_data(channel,get_index = False):
     training_series= []
     print "Checking..."
     res = preproc.check(channel)
     series = channel[2:]
     idx = itertools.chain.from_iterable(res[2])
     idx = sorted(list(idx))
     tr_idx = [0]
     tr_idx.extend(idx)
     tr_idx.append(len(series)-1)
     print "Training indeces: {idx}".format(idx =tr_idx)
     i=0
     pairs = []
     while i<=(len(tr_idx)-1):
          pair = []
          if tr_idx[i]!=tr_idx[i+1]:
               pair = [tr_idx[i]+1,tr_idx[i+1]-1]
          if pair:
               print pair
               training_pair = series[pair[0]:pair[1]]
               training_series.append(training_pair)
               pairs.append(pair)
          i+=2
     if get_index:
          return training_series, pairs
     else:
          return training_series

def get_idx_data(index):
     idx = []
     for part in index:
          idx.append(list(xrange(part[0],part[1])))
     return idx

def partition_set(data,ratio = 0.8):
     length = len(data)
     data_1= data[:int(length*ratio)]
     data_2 = data[int(length*ratio):]
     return data_1,data_2

def partition_data(data,ratio=0.8):
     tr_=[]
     te_=[]
     ratio=ratio
     for part in data:
          tr,te = partition_set(part,ratio=ratio)
          tr_.append(tr)
          te_.append(te)
     return tr_,te_

def transform(array, y=False):
     array =np.atleast_2d(array)
     if y==True:
          array = array.astype("float")
     array = array.T
     return array


