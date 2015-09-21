import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import itertools

## preprocess

def check(series, return_series = False):
     blank_cnt = 0
     cnt=0
     series_=np.array(series)
     new_series = []
     locs = []
     indeces=[]
     gap=0
     a = {}
     for i in series_:
          if (i =='-') or (i=='0'):
               i = np.nan
               #print "convert to 0"
               blank_cnt+=1
               if locs:
                    if cnt-locs[-1]!=1:
                         key = 'gap_'+str(gap)
                         gap+=1
                         a[key]=locs
                         locs=[]
               locs.append(cnt)
          else:
              i = float(i)
          new_series.append(i)
          cnt +=1
     key = 'gap_'+str(gap)
     a[key]=locs
     if blank_cnt == 0:
          print "No blank"
     else:
          print "converted {n} blanks".format(n=blank_cnt)
          print " {n} gaps: ".format(n=gap+1)
          gap_len = []
          for value in list(a.values()):
               indeces.append([value[0],value[len(value)-1]])
               gap_len.append(len(value))
               print "blank indeces: {index1} to {index2}. length = {length}".format(index1 = value[0],index2=value[len(value)-1], length=len(value))
          print blank_cnt/float(len(series_))
          #new_series=pd.Series(new_series)
          #return new_series
     if return_series == False:
          return blank_cnt, gap, indeces, gap_len, blank_cnt/float(len(series_))
     elif return_series == True:
          return new_series,blank_cnt, gap, indeces, gap_len, blank_cnt/float(len(series_))
     else:
          return 0,0,0,[0],0


def split_line(line):
     line = line[0].split('\t')[:-1]
     return line

def get_header(file_path,split=False):
     f= open(file_path,'r')
     reader = csv.reader(f)
     h = reader.next()
     if split:
          h = split_line(h)
     h = [str(i)[1:-1] for i in h]
     f.close()
     print "{f} includes columns: {h}".format(f = file_path,h=h)
     return h

def get_column(column, file_path,split = False):
     h = get_header(file_path,split)
     if column not in h:
          return 0
     else:
          print "Returning new column..."
          f=open(file_path,'r')
          new_col = []
          reader= csv.reader(f)
          idx = h.index(column)
          print idx
          for line in reader:
               if split:
                    line = split_line(line)
               if len(line)<=idx:
                    new_col.append('-')
                    print line
               else:
                    new_col.append(line[idx])
          print  "Done..."
     return new_col
          
