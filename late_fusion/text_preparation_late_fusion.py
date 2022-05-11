#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pickle
from collections import defaultdict
# warnings.filterwarnings('ignore')
import sklearn as sk
import pandas as pd
pd.options.display.max_colwidth = 1500
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from datetime import datetime
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import string
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import csv
import json
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split


# In[70]:


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# In[71]:


word_dict=dict()
word_dict['<PAD>'] = 10


# In[72]:


m = nn.Softmax(dim=2)


# In[326]:


class Dataset(Dataset):
    def __init__(self, data,label):
        'Initialization'
        dim_size = len(data[0][0])
        data_lengths= [len(frame) for frame in data]
        data_lengths_copy = [len(frame) for frame in data]
        data_lengths_copy.sort()
        pad_token = word_dict['<PAD>']
        print(data_lengths_copy)
        try:
            longest_frame = data_lengths_copy[-2]
        except:
            print("in except")
            longest_frame = data_lengths_copy[-1]
        b_s = len(data_lengths)
        padded_X = np.ones((b_s, longest_frame,dim_size)) * pad_token
        padded_Y = np.ones((b_s,longest_frame)) * pad_token

        print(padded_X.shape)
        for i, d_len in enumerate(data_lengths):
            sequence = data[i]
            sequence_y = label[i]
            if(d_len>longest_frame):
                continue
            
            padded_X[i, (longest_frame-d_len):] = sequence[:longest_frame]
            padded_Y[i,(longest_frame-d_len):] = sequence_y[:longest_frame]

        self.data = torch.Tensor(data)
        self.label = torch.LongTensor(label)
        self.original_data = data
        self.original_label = label
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X= self.data[index]
        y = self.label[index]
        return (X, y)


# In[327]:


csv_file_text = 'test_textual_feature_set_new_4.csv'


df = pd.read_csv(csv_file_text,header=None)
print(len(df), len(df[df[50] != '0']))
df = df[(df[50]== 'D') | (df[50]== 'Ah') | (df[50]== 'Ih') | (df[50]== 'Z') | (df[50]== 'W') | (df[50]== '0')  ]
unique_classes = df[50].unique().tolist()
unique_classes.remove('0')
indexes = [i for i in range(len(unique_classes))]
replace_dict = {unique_classes[i]:i for i in indexes}
replace_dict['0']=len(unique_classes)
df[[50]] = df[[50]].replace(replace_dict)
df_list  = df.values.tolist()
df_list_n_x = list()
df_l_sub_x=list()
df_list_n_y = list()
df_l_sub_y=list()
c =0
print("df_list", len(df_list))
actual = 0
for i in df_list:
    if(i[0]!=0):
        l = i[1:len(i)-1]
        l.insert(0,float(i[0]))
        df_l_sub_x.append(l)
        df_l_sub_y.append(i[len(i)-1])
        actual+=1
        c+=1
    else:
        if(len(df_l_sub_x)>0):
            actual += 1
            df_list_n_x.append(df_l_sub_x)
            df_list_n_y.append(df_l_sub_y)
        else:
            df_list_n_x.append(l)
            df_list_n_y.append(i[len(i) - 1])


        df_l_sub_x=list()
        df_l_sub_y=list()
print(len(df_list_n_x))


# In[ ]:





# In[328]:


len(df)


# In[329]:


len(df[df[50] != '0'])


# In[391]:


replace_dict = {'Ah':0, 'Ih':1, 'Z':2, 'W':3, 'D':4}
def get_data():
    csv_file_text = 'test_textual_feature_set_new_4.csv'
    
    df = pd.read_csv(csv_file_text,header=None)
    print(len(df), len(df[df[50] != '0']))
    df = df[(df[50]== 'D') | (df[50]== 'Ah') | (df[50]== 'Ih') | (df[50]== 'Z') | (df[50]== 'W') | (df[50]== '0')  ]
    print(len(df), len(df[df[50] != '0']), df[50].value_counts())
    print(len)
    replace_dict['0']=len(unique_classes)
    df[[50]] = df[[50]].replace(replace_dict)
    df_list  = df.values.tolist()
    df_list_n_x = list()
    df_l_sub_x=list()
    df_list_n_y = list()
    df_l_sub_y=list()
    c =0
    print("df_list", len(df_list))
    actual = 0

    for i in df_list:
        if(i[0]!=0):
            l = i[1:len(i)-1]
            l.insert(0,float(i[0]))
            df_l_sub_x.append(l)
            df_l_sub_y.append(i[len(i)-1])
            actual+=1
            c+=1
        else:
            if(len(df_l_sub_x)>0):
                actual += 1
                df_list_n_x.append(df_l_sub_x)
                df_list_n_y.append(df_l_sub_y)
                
            df_l_sub_x=list()
            df_l_sub_y=list()
    print(len(df_list_n_x), "--", actual)
    print(np.array(df_list_n_x).shape)
    return (df_list_n_x,df_list_n_y)



# In[392]:


def predict(model, dataloader):
    prediction_list = []
    test_data_list = []
    num_data_points = 0
    for data, target in dataloader:
        outputs = model(data)
        for i,dat in enumerate(outputs.data):
            num_data_points += 1
            if(target[0][i]!=torch.tensor(word_dict['<PAD>']) ):
                
                prediction_list.append(dat)
                test_data_list.append(target[0][i])
    print("num data points", num_data_points, len(dataloader))
    return (test_data_list,prediction_list)


# In[393]:


class GRUNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        tag_space = self.fc(out)
        tag_scores = m(tag_space)
        return tag_scores.view(batch_size*len(x[0]),-1)
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


# In[394]:


def main_process():
    test_set = get_data()
    #print("test set", test_set)
    test_set_x = test_set[0]
    test_set_y = test_set[1]
    predictions_all = []
    data_p = 0
    for i, l in enumerate(test_set_x):
        #print(i, l)
        print(np.array(l).shape)
        test_data = Dataset([l],[test_set_y[i]])

        data_p += 1
        test_loader_one = torch.utils.data.DataLoader( test_data, batch_size=1, num_workers=0)
        new_model = GRUNet(50,6,20,1)
        new_model.load_state_dict(torch.load('grunet.pt'))
        predictions = predict(new_model,test_loader_one)
        predictions_all.append(predictions)
    print(data_p, '*****', len(predictions_all))
    return predictions_all


# In[395]:


def write_file(predictions):
    final_t = []
    csv_file_text = 'late_fusion_text_results.csv'
    final_t = []
    for k in predictions:
        for i,element in enumerate(k[1]):
            print(element)
            t = list(element.numpy())
            t.append(k[0][i].item())
            final_t.append(t)
    final_t = pd.DataFrame(final_t)
    final_t.to_csv(csv_file_text, index = False)
    return final_t


# In[396]:


def print_results(predictions):
    pred = np.array(predictions[1])
    targ = np.array(predictions[0])
    print("---------- F1 Score -----------")
    print(metrics.f1_score(targ, pred,average='weighted'))
    print("---------- Accuracy -----------")
    print(metrics.accuracy_score(targ, pred))
    


# In[397]:


predictions = main_process()
final_t = write_file(predictions)
# print_results(predictions)


# In[398]:


len(final_t)


# In[399]:


len(predictions[5])


# In[400]:


predictions


# In[401]:


len(final_t)


# In[402]:


final_t


# In[403]:


predictions


# In[382]:


final_t = []
for k in predictions:
    for i,element in enumerate(k[1]):
        print(element)
        t = list(element.numpy())
        t.append(k[0][i].item())
        final_t.append(t)


# In[383]:


len(final_t)


# In[ ]:





# In[ ]:




