import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

## time series processing

def diff(pd_series, freq):
     series = pd_series
     diffed_series = (series - series.shift(freq))[freq:]
     return diffed_series


def forward_ewma(series, steps, window):
     cur_series = pd.Series(series)
     for i in range(steps):
          cur_series = cur_series.set_value(len(cur_series),np.asarray(pd.ewma(cur_series[-window:],span=100,))[-1])
     return cur_series[-steps:]

def fitfunc(p, x):
  return (p[0] * (1 - p[1] * np.sin(2 * np.pi / (90) * (x + p[2]))))

def residuals(p, y, x):
  return y - fitfunc(p, x)

def fit(tsdf):
  p0 = np.array([np.mean(tsdf), 1.0, 0.0])
  plsq, suc = leastsq(residuals(), p0)
  return plsq

def scale(array, range_int):
     new_array = []
     old_range = max(array)-min(array)
     new_range = range_int[1]-range_int[0]
     for i in array:
          i= (((i-min(array))*new_range)/old_range)+range_int[0]
          new_array.append(i)
     return new_array

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
     sig_ff = signal.filtfilt(b, a, sig)
     return sig_ff

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

