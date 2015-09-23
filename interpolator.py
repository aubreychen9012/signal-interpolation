import numpy as np
import pandas as pd
import math
from scipy import interpolate
import statsmodels.api as sm
from itertools import groupby
from operator import itemgetter
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import get_all_params
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
from lasagne.updates import adam
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import objective
from nolearn.lasagne import TrainSplit

import preproc
import gen_add_feat
import clf
import signal_preproc

## neuralnet interpolator
## input: tr, add_feat_tr, add_feat_pred
## output: interpolated data for gap

def downsampling(data, threshold):
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

def upsampling(data,upscale):
    x = range(len(data))
    newx = [i/float(upscale)*x[-1] for i in range(upscale)]
    f = interpolate.interp1d(x,data)
    new_data = f(newx)
    new_data= new_data[1:-1]
    st = [data[0]]
    st.extend(new_data)
    st.append(data[-1])
    return st

def build_feat(data, add_feat,window_length=20):
    ## use other available channels
    feat_mean = np.nan_to_num(pd.rolling_mean(data,5))
    feat_mean = signal_preproc.rolling_window(feat_mean, window_length)
    feat_std = np.nan_to_num(pd.rolling_std(data,5))
    feat_std = signal_preproc.rolling_window(feat_std, window_length)
    feat_diff = np.append(0,np.diff(data))
    feat_diff = signal_preproc.rolling_window(feat_diff,window_length)
    feat_grad = [np.gradient(i) for i in signal_preproc.rolling_window(data,window_length)]
    feat_orig = signal_preproc.rolling_window(data,window_length)
    feat_2ndgrad = [np.gradient(i,2) for i in signal_preproc.rolling_window(data,window_length)]
    feat_gmean = [clf.RunningGMean(i) for i in signal_preproc.rolling_window(data,window_length)]
    feat_ii = signal_preproc.rolling_window(add_feat, window_length)
    feat = np.hstack((feat_orig,feat_mean,feat_std,feat_diff,feat_grad,feat_2ndgrad,feat_gmean,feat_ii))
    feat = feat.reshape(-1,8,window_length)
    feat = feat.astype('float32')
    return feat

def get_target(data,window_length=20):
    target = signal_preproc.labels(signal_preproc.rolling_window(data,window_length))
    target = np.atleast_2d(np.asarray(target)).T
    target = target.astype('float32')
    return target

class NeuralnetInterpolator:
     def __init__(self,tr,add_feat_tr,add_feat_pred,window_length,bool_var):
          self.tr = tr
          self.add_feat_tr = add_feat_tr
          self.add_feat_pred = add_feat_pred
          self.window_length = window_length
          self.trend = bool_var

     def est_trend(self,tr):
          data = tr
          x = range(len(data))
          lowess = sm.nonparametric.lowess(data, x, frac=0.0005)
          return lowess[:,1]

     def trend_clf(self,tr,len_add_feat_pred):
          length = len(len_add_feat_pred)/10
          trend = self.est_trend(tr)
          trend_ = downsampling(trend,len(trend)/10)
          if len(trend_)>500:
              trend_ = trend_[:500]
          res = sm.tsa.arma_order_select_ic(trend_, max_ar = 6, max_ma = 6,ic=['aic', 'bic'], trend='c')
          params = res['bic_min_order']
          arma_model = sm.tsa.ARMA(trend_, params).fit
          predict_trend = arma_model.predict(len(trend_)-10, len(trend_)+length, dynamic=True)
          predict_trend = upsampling(predict_trend,upscale = self.add_feat_pred)
          return predict_trend

     def fit(self,tr,add_feat_tr):
          ## if trend exists, remove trend
          if self.trend ==1:
              trend = self.est_trend(tr)
              tr = tr-np.asarray(trend)
          layers0=[
               ## 2 layers with one hidden layer
               (InputLayer, {'shape': (None,8,self.window_length)}),
               (DenseLayer, {'num_units': 8*self.window_length}),
               (DropoutLayer, {'p':0.3}),
               (DenseLayer, {'num_units': 8*self.window_length/3}),
               ## the output layer
               (DenseLayer, {'num_units': 1, 'nonlinearity': None}),
          ]
          feats = build_feat(tr, add_feat_tr, window_length=self.window_length)
          print feats.shape
          feat_target = get_target(tr,window_length=self.window_length)
          print feat_target.shape
          net0 = NeuralNet(
               layers=layers0,
               max_epochs=400,
               update=nesterov_momentum,
               update_learning_rate=0.01,
               update_momentum=0.9,
               verbose=1,
               regression=True,
          )
          net0.fit(feats[:-1],feat_target)
          return net0,feats,feat_target

     def validate(self,tr,add_feat_tr):
         feats = build_feat(tr, add_feat_tr, window_length=self.window_length)
         feat_target = get_target(tr,window_length=self.window_length)
         net0,feats,feat_target = self.fit(tr,add_feat_tr)
         pred = clf.predict_feat(feats[0],net0,add_feat_tr,new_append_value=0,\
                                 window_length = self.window_length,lags = len(add_feat_tr)-1)
         mse_error = clf.MSE(pred,feat_target.reshape(len(pred),).tolist())
         return mse_error

     def interpolate(self):
         if len(self.add_feat_pred)/10<=2:
             tr = np.asarray(self.tr)
             add_feat_tr = np.asarray(self.add_feat_tr)
             add_feat_pred = np.asarray(self.add_feat_pred)
         else:
             tr = np.asarray(downsampling(self.tr,len(self.tr)/10)).T[1]
             add_feat_tr = np.asarray(downsampling(self.add_feat_tr,len(self.add_feat_tr)/10)).T[1]
             add_feat_pred = np.asarray(downsampling(self.add_feat_pred,len(self.add_feat_pred)/10)).T[1]
         print len(tr),len(add_feat_tr),len(add_feat_pred)
         net0,feats,feat_target=self.fit(tr,add_feat_tr)
         pred_res = clf.predict_feat(feats[-1],net0,add_feat_pred,new_append_value=0,\
                                     window_length = self.window_length,lags = len(add_feat_pred))
         if len(self.add_feat_pred)/10>2:
             pred_res = upsampling(pred_res,upscale = len(self.add_feat_pred))
         if self.trend==1:
             pred_res += self.trend_clf(tr,len(add_feat_pred))
         return pred_res
         ## if MSE is nan, return 0 and pass it to copypaste

## if neuralnet interpolator doesn't work well, use copypaste interpolator 
## to get data from before and after the gap to fill in the gap
class CopyPasteInterpolator:
    def __init__(self,sig):
        if not isinstance(sig,np.ndarray):
            sig = np.asarray(sig)
        idx_nan_group = []
        idx_nan = np.isnan(sig)
        idx = np.arange(len(sig))
        idx_nan_idx = idx[idx_nan]
        for k, g in groupby(enumerate(idx_nan_idx), lambda (i,x):i-x):
            idx_nan_group.append(map(itemgetter(1),g))
        self.gaps= idx_nan_group
        if self.gaps[0][0]==0:
            self.gaps = self.gaps[1:]
        self.sig = sig
    def interpolate(self):
        idx_nan_group = self.gaps
        for group in idx_nan_group:
            part_fw=[]
            l = len(group)
            if (l>1) and (l<len(self.sig)/2):
                if group[0]<=l/2:
                    part_bw = self.sig[group[-1]+1:group[-1]+1+l]
                elif group[-1]+l/2+2>len(self.sig):
                    part_bw = self.sig[group[0]-l:group[0]]
                elif l%2:
                    part_bw=self.sig[group[0]-l/2:group[0]]
                    part_fw=self.sig[group[-1]+1:group[-1]+l/2+2]
                elif not l%2:
                    part_bw=self.sig[group[0]-l/2:group[0]]
                    part_fw=self.sig[group[-1]+1:group[-1]+l/2+1]
            elif l>=len(self.sig)/2:
                part_bw=self.sig[:group[0]]
                part_fw=self.sig[group[-1]+1:]
                #sig[:group[0]+len(part_bw)]=part_bw
                #sig[group[-1]-len(part_fw):]=part_fw
            elif l==1:
                part_bw = self.sig[group[0]-1] if group[0]!=0 else self.sig[group[0]+1]
            if len(part_fw)>0:
                part_bw= np.append(part_bw, part_fw)
                if np.isnan(part_bw).any():
                    part_bw = CopyPasteInterpolator(part_bw).interpolate()
                if len(part_bw)!=len(group):
                    print 'length unmatch'
                    break
            self.sig[group[0]:group[0]+l] = part_bw
        new_sig = np.copy(self.sig)
        return new_sig

## main interpolator
class SignalInterpolator:
     def __init__(self,col,pf,window_length,bool_var):
          self.col = col
          self.pf = pf
          self.window_length = window_length
          self.bool_var = bool_var

     def interpolate(self):
          sig_main = preproc.get_column(self.col,self.pf)
          res = preproc.check(sig_main[2:],return_series=1)
          if res[-1]>=0.9:
              return sig_main
          sig_main_gaps = res[3]
          sig_array = res[0]
          active_chl = self.col
          for gap in sig_main_gaps:
               rng = gap
               data = gen_add_feat.add_feat(rng,active_chl,self.pf)
               if data!=0:
                    tr, add_feat_tr,add_feat_pred = data[0],data[1],data[2]
                    nn = NeuralnetInterpolator(tr,add_feat_tr,add_feat_pred,self.window_length,self.bool_var)
                    new_sig = nn.interpolate()
                    sig_array[rng[0]:rng[-1]] = new_sig
               else:
                    pass
          sig_interpolated = sig_array
          return sig_interpolated



     
          
