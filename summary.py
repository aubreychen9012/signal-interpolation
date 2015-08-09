## quick stats on gaps
import csv
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append('/Users/aubrey9012/Desktop')
import preproc
import training

def summarize(file_name):
     file_path = '/Users/aubrey9012/Downloads/medical_ICU-features/data/mimic/'+str(file_name)+'.csv'
     output_path = '/Users/aubrey9012/Downloads/medical_ICU-features/data/output/summary_of_'+str(file_name)+'.csv'
     f = open(file_path, 'r')
     output = open(output_path, 'wt')
     writer = csv.writer(output)
     reader = csv.reader(f)
     h = reader.next()
     header = [str(i)[1:-1] for i in h]
     print header
     f.close()
     f = open(file_path, 'r')
     key = ['black_cnt', 'gaps', 'gap_indeces', 'missing_value_proportion']
     for col in header:
          writer.writerow([col])
          writer.writerow(key)
          sig = preproc.get_column(col, file_path)
          sig = sig[2:]
          res = preproc.check(sig, return_series =False)
          writer.writerow(res)
     f.close()
     output.close()
     return

if __name__ == "__main__":
     file_names = [3642023,3655233,3656395,3668415]    ## put files name to summarize
     Parallel(n_jobs =2)(delayed(summarize)(file_name) for file_name in file_names)
