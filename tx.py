__author__ = 'aubrey9012'

import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/aubrey9012/Desktop')
import preproc
import interpolator

col = 'ICP'
file_name = '3270954'
pf = '/Users/aubrey9012/Downloads/medical_ICU-features/data/mimic/'+str(file_name)+'.csv'

si = interpolator.SignalInterpolator(col,pf,40)
si_ = si.interpolate()
