import numpy as np
import pandas as pd
import math
from scipy import interpolate
import statsmodels.api as sm
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
    data_ = data[1:-1]
    x = range(len(data))
    newx = [i/float(len(upscale)-2)*x[-1] for i in range(upscale-2)]
    f = interpolate.interp1d(x,data_)
    new_data = f(newx)
    st = [data[0]]
    st.extend(new_data)
    st.append(data[-1])
    return st

def build_feat(data, add_feat,window_length=20):
    data = downsampling(data,len(data)/10)
    data = np.asarray(data).T[1]
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
    target = signal_preproc.labels(signal_preproc.rolling_window(data,window_length=20))
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

     def fit(self, tr,add_feat_tr):
          ## if trend exists, remove trend
          if self.trend ==1:
              trend = self.est_trend(tr)
              tr = tr-np.asarray(trend)
          layers=[
               (InputLayer, {'shape': (None,8, 2*self.window_length)}),
               (DenseLayer, {'num_units': 8*2*self.window_length}),
               (DropoutLayer, {'p':0.3}),
               (DenseLayer, {'num_units': 8*2*self.window_length/3}),
               ]
          feats = build_feat(tr, add_feat_tr, self.window_length)
          feat_target = get_target(tr,self.window_length)
          net0 = NeuralNet(
               layers=layers,
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
         feats = build_feat(tr, add_feat_tr, self.window_length)
         feat_target = get_target(tr,self.window_length)
         net0,feats,feat_target = self.fit(tr,add_feat_tr)
         pred = clf.predict_feat(feats[0],net0,add_feat_tr,new_append_value=0,\
                                 window_length = self.window_length,lags = len(add_feat_tr)-1)
         mse_error = clf.MSE(pred,feat_target.reshape(len(pred),).tolist())
         return mse_error

     def interpolate(self):
         tr = np.asarray(downsampling(self.tr,len(self.tr)/10)).T[1]
         add_feat_tr = np.asarray(downsampling(self.add_feat_tr,len(self.add_feat_tr))).T[1]
         add_feat_pred = np.asarray(downsampling(self.add_feat_pred,len(self.add_feat_pred))).T[1]
         net0,feats,feat_target=self.fit(tr,add_feat_tr)
         pred_res = clf.predict_feat(feats[-1],net0,add_feat_pred,new_append_value=0,\
                                     window_length = self.window_length,lags = len(add_feat_pred)-1)
         pred_res = upsampling(pred_res,upscale = len(self.add_feat_pred))
         if self.trend==1:
             pred_res += self.trend_clf(tr,len(add_feat_pred))
         return pred_res

class CopypasteInterpolator:
     pass

class SignalInterpolator:
     def __init__(self,col,pf,window_length):
          self.col = col
          self.pf = pf
          self.window_length = window_length

     def interpolate(self):
          sig_main = preproc.get_column(self.col,self.pf)
          res = preproc.check(sig_main[2:],return_series=1)
          sig_main_gaps = res[3]
          sig_array = res[0]
          active_chl = self.col
          for gap in sig_main_gaps:
               rng = gap
               data = gen_add_feat.add_feat(rng,active_chl,self.pf)
               if data!=0:
                    tr, add_feat_tr,add_feat_pred = data[0],data[1],data[2]
                    nn = NeuralnetInterpolator(tr,add_feat_tr,add_feat_pred,self.window_length)
                    new_sig = nn.interpolate
                    sig_array[rng[0]:rng[-1]] = new_sig
               else:
                    pass
          sig_interpolated = sig_array
          return sig_interpolated



     
          
