import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import collections

## time series processing
##################################
# differencing, or np.diff
def diff(pd_series, freq):
     series = pd_series
     diffed_series = (series - series.shift(freq))[freq:]
     return diffed_series

# fit with forward ewma
def forward_ewma(series, steps, window):
     cur_series = pd.Series(series)
     for i in range(steps):
          cur_series = cur_series.set_value(len(cur_series),np.asarray(pd.ewma(cur_series[-window:],span=100,))[-1])
     return cur_series[-steps:]

##################################
# fit simple sine curve
def fitfunc(p, x):
  return (p[0] * (1 - p[1] * np.sin(2 * np.pi / (90) * (x + p[2]))))

def residuals(p, y, x):
  return y - fitfunc(p, x)

def fit(tsdf):
  p0 = np.array([np.mean(tsdf), 1.0, 0.0])
  plsq, suc = leastsq(residuals(), p0)
  return plsq

##################################

## scale array to a certain range
def scale(array, range_int):
     new_array = []
     old_range = max(array)-min(array)
     new_range = range_int[1]-range_int[0]
     for i in array:
          i= (((i-min(array))*new_range)/old_range)+range_int[0]
          new_array.append(i)
     return new_array

##################################

## low pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# filtfilt filtering
def filt_filt_filtering(data,cutoff,fs,order):
     b, a = butter_lowpass(cutoff, fs, order=order)
     sig_ff = signal.filtfilt(b, a, data)
     return sig_ff

###################################

# read sig in rolling windows of defined size
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# label each rolling window
def labels(rolling_series):
    len_roll = len(rolling_series)
    labels_list = []
    for i in range(1,len_roll):
        labels_list.append(rolling_series[i][-1])
    return labels_list

# reshape rolling series into an array to pass on regressor
def rolling_to_train(rolling_series):
    train_rl = rolling_series[:-1]
    return train_rl

# pop the first item of x and append y to x
def shift(x,y):
     x = collections.deque(x)
     x.popleft()
     shift_x = np.append(x,y)
     return shift_x     

def predict_rolling(clf, x, y_upper):
     pred_y=[]
     window = collections.deque(x)
     y0 = clf.predict(window)
     pred_y.append(y0.tolist())
     while len(pred_y)<y_upper:
          window = shift(window,y0)
          y0 = clf.predict(np.asarray(window))
          pred_y.append(y0.tolist())
     return pred_y

def predict_rolling_nn(clf, x, y_upper):
     pred_y=[]
     window = x
     y0 = clf.predict(window)[0]
     pred_y.append(y0.tolist())
     while len(pred_y)<y_upper:
          window = shift(window.tolist()[0][:-2],y0)
          window = np.append(window,[np.mean(window),np.std(window)])
          window = np.atleast_2d(window)
          y0 = clf.predict(window)[0]
          pred_y.append(y0.tolist())
     return pred_y

     
     

