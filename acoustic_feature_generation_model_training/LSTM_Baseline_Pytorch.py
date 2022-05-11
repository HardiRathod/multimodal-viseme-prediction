#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## CHANGE the number of layers and layer size here.

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        #predictions = self.linear(lstm_out)
        return predictions


# In[ ]:


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#criterion = 


# In[ ]:


class T2DV2Dataset(Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        # self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.train_x)
    
    def __getitem__(self, index):
        # print(len(self.train_y), index, self.train_y[13])
        return torch.tensor(self.train_x[index]), torch.tensor(self.train_y[index])


# In[ ]:


def train_model_function(n_epochs, train_x, train_y, test_x, test_y, batch_size, learning_rate, input_data_size, number_of_classes, model_name):
    print("-------------Loading Dataset----------")
    train_dataset = T2DV2Dataset(train_x, train_y)
    test_dataset = T2DV2Dataset(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    model_ff = LSTM(20, 100, number_of_classes)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("-----------Starting the Model Training -------------")
    valid_loss_min = np.Inf 
    for epoch in range(n_epochs):

        train_loss = 0.0
        valid_loss = 0.0
        print("Starting epoch")
        model_ff.train() # prep model for training
        for data, target in train_loader:
            print(len(data))
            model_ff.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            target = target.type(torch.LongTensor)
            #print(data.shape, hidden.shape)
            
            output = model_ff(data.float())
        
            # output = torch.argmax(output, dim=1)
            #target = target.reshape(target.shape[0])
            #print(output, target)
            loss = criterion(output, target).clone()
            print(loss)
            loss.backward()
            train_loss += loss.item()*data.size(0)
            optimizer.step()
            # hidden = hidden_1
               
        model_ff.eval()
        print("Evaluating")
        for data, target in test_loader:
            model_ff.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
            torch.autograd.set_detect_anomaly(True)
            target = target.type(torch.LongTensor)
            # print(data.shape, hidden.shape)
            target = target.reshape(target.shape[0])
            output = model_ff(data.float())
            loss = criterion(output, target) 
            valid_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(test_loader.dataset)


        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, 
            train_loss,
            valid_loss
            ))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model_ff.state_dict(), model_name)
            valid_loss_min = valid_loss


# In[ ]:


## Running the above model for MFCC feature set
data = pd.read_csv('/content/drive/Shareddrives/CS535 Project/mfcc_feature_set.csv', header = None)


# In[ ]:


data.head(10)


# In[ ]:


data = data[data[20]!= 'Oy']
data = data.dropna()


# In[ ]:



unique_classes = data[20].unique().tolist()
indexes = [i for i in range(len(unique_classes))]
replace_dict = {unique_classes[i]:i for i in indexes}
data[[20]] = data[[20]].replace(replace_dict)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[[i for i in range(20)]], data[[20]].values.ravel(), test_size=0.25, random_state=0, stratify = data[[20]])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
#Print the shapes
X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val)


# In[ ]:


X_train = X_train.to_numpy().astype('float32')
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.to_numpy().astype('float32')
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))


# In[ ]:


num_workers = 0
train_model_function(5, X_train, y_train, X_val, y_val, 100, 0.001, 20, len(unique_classes), 'model_name')


# In[ ]:


X_train


# In[ ]:




