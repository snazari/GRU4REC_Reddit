"""

Sam Nazari
Systems & Technology Research
May 2018

"""

import numpy as np
import pandas as pd
import datetime as dt
#import dask.dataframe as dd

def string2int(df: pd.DataFrame)->pd.DataFrame:
    """ This function takes strings and assigns a unique integer to the string. """
    sr = list(set(df.SubReddit))
    sri= list(range(0,len(sr)))
    dict_sr=dict(zip(sr,sri))
    ds1 = df['SubReddit'].map(dict_sr)
    df=df.join(ds1,rsuffix='ID')
    sr = list(set(df.UserName))
    sri=list(range(0,len(sr)))
    dict_sr=dict(zip(sr,sri))
    ds1 = df['UserName'].map(dict_sr)
    df=df.join(ds1,rsuffix='Session')
    df.drop(['UserName','SubReddit'],inplace=True,axis=1)
    df.columns = ['Time','SubRedditID','Session']
    return df

def read_proc_clean(x: list,name: list)->list:
    """ This function reads csv files from the list x, processes it and returns a list of dataframes."""
    dfL = list()
    for s in x:
        dfL.append(pd.read_csv(s,names=name,usecols=[0,1,3]))
    l_new = list()
    for k in dfL:
        k.dropna(inplace=True)
        k = k[k.UserName != '[deleted]']
        k.Time = pd.to_datetime(k['Time'])
        k = string2int(k)
        l_new.append(k)
    dfL = l_new
    return dfL

def group_sort(dl: list)->list:
    l_new = list()
    for x in dl:
        x=x.groupby('Session').filter(lambda x: len(x)>=3)
        x=x.set_index(['Session','Time'])
        x=x.sort_index(level=0)
        l_new.append(x)
    dl = l_new
    return dl

if __name__ == "__main__":
    file_one = 'reddit_01_17_posts.csv'
    file_two = 'reddit_02_17_posts.csv'
    file_three='reddit_03_17_posts.csv'
    proc_file ='reddit_dataset.csv'
    fList = [file_one]
    names = ['Time','UserName','SubReddit']

    dL = read_proc_clean(fList,names)
    dL = group_sort(dL)
    df = dL[0]
    df.to_csv(proc_file)