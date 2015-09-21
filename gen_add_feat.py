import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/aubrey9012/Desktop')
import preproc


def discard_add_feat(chl,rng):
     if np.isnan(chl[(rng[0]-10000-1):(rng[0]-1)]).any():
          return 1
     else:return 0

def max_add_feat(active_chl,rng,pf):
    rng = rng
    min_per=1
    addfeat_data = 0

    header = preproc.get_header(pf)
    header_non_act = np.copy(header).tolist()
    header_non_act.remove(active_chl)
    header_non_act.remove('Elapsed time')
    for chl in header_non_act:
        sig_add = preproc.get_column(chl,pf)[2:]
        sig_add = preproc.check(sig_add,return_series=1)[0]
        chl_add = sig_add[rng[0]:rng[-1]]
        if not discard_add_feat(sig_add,rng):
            idx_isnan = np.isnan(chl_add)
            idx = np.arange(len(chl_add))
            idx_nan = idx[idx_isnan]
            percent = len(idx_nan)/float(len(idx))
            if min_per ==1:
                min_per=percent
                min_chl = sig_add
            elif percent<=min_per:
                min_per = percent
                min_chl = sig_add
            if min_per==0:
                break
    addfeat_data = min_chl[rng[0]-50000:rng[0]]
    if addfeat_data!=0:
        if np.isnan(addfeat_data).any():
            idx = np.arange(len(addfeat_data))
            idx_isnan = np.isnan(addfeat_data)
            idx = idx[idx_isnan].tolist()
            idx_start = idx[-1]
            addfeat_data = addfeat_data[idx_start:]
    return chl_add, addfeat_data

def gen_tr_data(rng,active_chl_array):
    tr = active_chl_array[rng[0]-50000-1:rng[0]-1]
    if np.isnan(tr).any():
        idx_isnan = np.isnan(tr)
        idx_start = idx_isnan[-1]
        tr = tr[idx_start:]
        if len(tr)>1e4:
            return tr
        else:
            return 0

def add_feat(rng, active_chl,pf):  ## name of active channel
    active_chl= active_chl
    rng = rng
    pf = pf
    tr=0
    add_feat_pred,add_feat_tr = max_add_feat(active_chl,rng,pf)
    print add_feat_pred,add_feat_tr
    active_chl_array = preproc.get_column(active_chl,pf)[2:]
    active_chl_array = preproc.check(active_chl_array,return_series=1)[0]
    tr = gen_tr_data(rng,active_chl_array)
    if tr!=0 and add_feat_tr!=0:
        tr_len = min(len(tr),len(add_feat_tr))
        tr = tr[-tr_len:]
        add_feat_tr = add_feat_tr[-tr_len:]
        return tr, add_feat_tr, add_feat_pred
    else:
        return 0
     
