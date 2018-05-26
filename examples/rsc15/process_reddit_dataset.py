"""

Sam Nazari
Systems & Technology Research
May 2018

"""
import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import datetime as dt
import time
import gru4rec
import evaluation
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
    df.columns = ['TimeStr','ItemId','SessionId']
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
        #k.Time = pd.to_datetime(k['Time'])
        #k.Time = k['Time'].astype(float)
        k.Time = k.Time.map(lambda x: time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(x)))
        k = string2int(k)
        l_new.append(k)
    dfL = l_new
    return dfL

def group_sort(dl: list)->list:
    l_new = list()
    for x in dl:
        x=x.groupby('SessionId').filter(lambda x: len(x)>=3)
        x=x.set_index(['SessionId','TimeStr'])
        x=x.sort_index(level=0)
        l_new.append(x)
    dl = l_new
    return dl

if __name__ == "__main__":
    file_one = 'reddit_01_17_posts.csv'
    file_two = 'reddit_02_17_posts.csv'
    file_three='reddit_03_17_posts.csv'
    proc_file ='reddit_dataset.csv'
    PATH_TO_TRAIN = 'reddit_train_full.txt'
    PATH_TO_TEST = 'reddit_test.txt'

    names = ['Time','UserName','SubReddit']

    fList = [file_one]
    dL = read_proc_clean(fList,names)
    dL = group_sort(dL)
    df = dL[0]
    df.to_csv(proc_file)
    
    print('Reading data.')
    data = pd.read_csv(proc_file, sep=',', header=0, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
    data.columns = ['SessionId', 'TimeStr', 'ItemId']
    data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').timestamp())
    data.drop('TimeStr',axis=1,inplace=True)
    
    print('Mini batching')
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_test = session_max_times[session_max_times >= tmax-86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),train.ItemId.nunique()))
    train.to_csv('reddit_train_full.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv('reddit_test.txt', sep='\t', index=False)
    
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    session_valid = session_max_times[session_max_times >= tmax-86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv('reddit_train_tr.txt', sep='\t', index=False)
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv('reddit_train_valid.txt', sep='\t', index=False)
    
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})
    
    #Reproducing results from "Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    
    print('Training GRU4Rec with 100 hidden units')    
    
    gru = gru4rec.GRU4Rec(loss='top1', final_act='tanh', hidden_act='tanh', layers=[100], batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, time_sort=False)
    gru.fit(data)
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    
    
    #Reproducing results from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
    
    #print('Training GRU4Rec with 100 hidden units')

    #gru = gru4rec.GRU4Rec(loss='bpr-max-0.5', final_act='linear', hidden_act='tanh', layers=[100], batch_size=32, dropout_p_hidden=0.0, learning_rate=0.2, momentum=0.5, n_sample=2048, sample_alpha=0, time_sort=True)
    #gru.fit(data)
    
    #res = evaluation.evaluate_sessions_batch(gru, valid, None)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))