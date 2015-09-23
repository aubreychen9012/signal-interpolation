__author__ = 'aubrey9012'

import numpy as np
import pandas as pd
import pickle
import sys

sys.path.append('/Users/aubrey9012/Desktop')
import preproc
import interpolator

file_name = '3270954'
pf = '/Users/aubrey9012/Downloads/medical_ICU-features/data/mimic/'+str(file_name)+'.csv'
header = preproc.get_header(pf)
header.remove('Elapsed time')

for col in header:
    if col != 'ICP':
        si = interpolator.SignalInterpolator(col,pf,40,0)
    else:
        si = interpolator.SignalInterpolator(col,pf,40,1)
    si_  = si.interpolate()
    if si_!=0:
        cpi = interpolator.CopyPasteInterpolator(si_)
        si_full = cpi.interpolate()
        dump_name= str(col)+'.p'
        pickle.dump(si_,open(dump_name, "wb" ))
    if si_==0:
        pickle.dump(si_,open(dump_name,"wb"))




output_fname = '/Users/aubrey9012/Desktop/'+str(file_name)+'p.csv'
output = open(output_fname,'wb')
h = ['Elapsed time']
h.extend(header)
writer = csv.writer(output)
writer.writerow(h)
unit = ['seconds','mV','mV','mV','mV','mV','pm','NU','mmHg','mmHg']
writer.writerow(unit)

for i in range(len(time)):
    timestamp = [time[i]]
    timestamp.extend([sig_i.popleft(),sig_ii.popleft(),sig_iii.popleft(),sig_avr.popleft(),sig_v.popleft(),\
                      sig_resp.popleft(),sig_pleth.popleft(),sig_abp.popleft(),sig_icp.popleft()])
    writer.writerow(timestamp)
output.close()

f2 = open(str(col)+'.p','r')
a = csv.reader(f2)

with open(output_fname,'r+') as f1:
    for line in f1:
        line = line.strip()
        line[1] = a.next()[0].split('F')[-1]

        f1.write(line)



for i in range(len(time)):
    timestamp = [time[i]]
    for col in header:
        l=[]
        val = reader(col).add()
        if len(l)>0:
            l.append(val)
        else:
            l = [val]
    timestamp.extend(l)
    print timestamp
    writer.writerow(timestamp)

output.close()







