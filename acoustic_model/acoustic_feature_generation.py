#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas
import xml.etree.ElementTree as ET
import os
import csv
import re
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


# In[20]:


def read_csv(file_name):
    file = open(file_name)
    df = pandas.read_csv(file_name)
    return df

    


# In[21]:


def read_bml(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    return root
    


# In[31]:


def read_viseme(root):
    viseme_l = list()
    for child in root:
        viseme = list()
#         print(child.tag)
        if(child.tag == "lips"):
#             print(True)
            viseme.append(child.attrib['viseme'])
            viseme.append(child.attrib['start'])
            viseme.append(child.attrib['end'])
            viseme_l.append(viseme)
    return viseme_l
            


# In[32]:


def read_transcript(root,end_time):
    for child in root:
        if(child.tag=='speech'):
            for child2 in child:
                if(child2.tag == 'text'):
                    texttag = child2
                    break
    transcript_l = list()
    next_t =0
    for i,sync in enumerate(texttag):
        transcript = list()
        for t in sync:
            print(t)
        if sync.text is None : 
            transcript.append('000')
        else:    
            transcript.append(sync.text)
#         transcript.append(prev)
        transcript.append(sync.attrib['time'])
        if(i<len(texttag)-1):
            transcript.append(texttag[i+1].attrib['time'])
        else:
            transcript.append(end_time)
        
        prev = sync.attrib['time']
        transcript_l.append(transcript)
    return transcript_l


# In[33]:


def get_mfcc(filename,viseme_extract):
    mfcc_l = list()
    for rec in viseme_extract:
        try:
            mfcc = create_mfcc(filename,float(rec[1]),float(rec[2])).tolist()
            mfcc.append(rec[0])
            mfcc_l.append(mfcc)
        except:
            pass
    return mfcc_l


# In[34]:


def create_mfcc(audio_path, start_ms, end_ms):
    start = start_ms/100
    end = end_ms/100
    data, sample_rate = lb.load(audio_path, offset = start, duration = end - start)
#     lb.display.waveplot(data)
#     print(type(data), type(sample_rate))
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
#     librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
    mfccScaled = np.mean(mfcc.T, axis=0)
    return mfccScaled


# ## Modify files 

# In[35]:


read_directory = '/Users/bidishadasbaksi/Docs_no_icloud/Documents – Bidisha’s MacBook Pro/ Personal_docs Intuit Laptop/USC Journey/Spring 2022/CSCI-535 Multimodal/Project/segmented/viseme'
write_directory = '/Users/bidishadasbaksi/Docs_no_icloud/Documents – Bidisha’s MacBook Pro/ Personal_docs Intuit Laptop/USC Journey/Spring 2022/CSCI-535 Multimodal/Project/segmented/viseme_modified_new'


# ## Run code below only to modify viseme files

# In[36]:


# for filename in os.listdir(read_directory):
#     file_path_read = os.path.join(read_directory, filename)
#     file_path_write = os.path.join(write_directory, filename)
#     f_r = open(file_path_read,'r+')
#     f_w = open(file_path_write,'w+')
#     lines = f_r.readlines()
#     for line in lines:
#         if(re.match('.*sync.*',line)):
#             line= line.replace('/','')
#             line = line.replace('\n','')
#             line = line + '</sync>\n'
#             f_w.write(line)
#         else:
#             f_w.write(line)
#     f_r.close()
#     f_w.close()


# ## Read BML files

# In[37]:


read_viseme_mod = 'viseme_modified_2'
audio_dir = 'G:/Shared drives/CS535 Project/data/raw_audio'
csv_file = '/Users/bidishadasbaksi/Docs_no_icloud/Documents – Bidisha’s MacBook Pro/ Personal_docs Intuit Laptop/USC Journey/Spring 2022/CSCI-535 Multimodal/Project/segmented/mfcc_feature_set.csv'



# In[38]:


# csv_p = open(csv_file,'a')
# csv_write = csv.writer(csv_p)


# In[39]:


viseme_set = list()
transcript_set = list()
viseme_feature_set = list()
filename_set = list()


# In[15]:


def feature_gneration(audio_gen):
    for filename in os.listdir(read_viseme_mod):
        file_path_read = os.path.join(read_viseme_mod, filename)
    #     print(file_path_read)
        f_r = open(file_path_read,'r',  encoding = 'utf-8-sig' )
        root = read_bml(f_r)
        viseme_extract = read_viseme(root)
        f_split = filename.split("_")
        audio_session = f_split[0][4]
        if(len(f_split)==4):
            audio_file_dir = f_split[0]+"_"+f_split[1]+"_"+f_split[2]
        else:
            audio_file_dir = f_split[0]+"_"+f_split[1]
        audio_file_name = filename.split(".")[0]+".wav"
        audio_path = audio_dir + "/Session" + audio_session + "/sentences/wav/"+audio_file_dir+"/"+audio_file_name
    #     print(f_r)
        if(audio_gen):
            mfcc_= get_mfcc(audio_path,viseme_extract)
            with open(csv_file,'a') as f:
                csv_write = csv.writer(f)
                csv_write.writerows(mfcc_)
                csv_write.writerow(list('\n'))
            viseme_feature_set.append(mfcc_)
        viseme_set.append(viseme_extract)
    #     print(viseme_extract[len(viseme_extract)-1][2])
        transcript_set.append(read_transcript(root,viseme_extract[len(viseme_extract)-1][2]) )


# In[16]:


feature_gneration(False)


# In[17]:


viseme_set


# In[18]:


transcript_set


# ## Total number of viseme classes

# In[40]:


vis_dict = dict()


# In[41]:


for i in viseme_set:
    for j in i:
        if(j[0] not in vis_dict):
            vis_dict[j[0]] = 1


# In[42]:


len(vis_dict)


# ### Total no of unique words  - Word2Vec Vectors

# In[43]:


import gensim.downloader as api
import gensim.models
from gensim.test.utils import datapath
from gensim import utils


# In[ ]:


check_mapping = dict()
count = 0 


# In[ ]:

'''
for i in word2vec_l:
    if i is not None : 
        if i in model.wv:
            check_mapping[i] = model.wv[i]
        else:
            print(i)


# In[ ]:


word2vec_dict


# In[ ]:



model = Word2Vec(word2vec_l, min_count=2)

# ## Merge Set

# In[63]:


words_list = []
viseme_text_features = []


# In[45]:


def textual_features(text_map,viseme_map):
    merge_set = []
    for i in range(0, len(viseme_map)):
        tmp = []   
        start_time = float(viseme_map[i][1])
        for j in range(0, len(text_map)-1):
            if float(text_map[j][1]) <= start_time < float(text_map[j+1][1]):
                tmp.append((viseme_map[i][0], text_map[j][0], viseme_map[i][1], viseme_map[i][2]))
        if not tmp:
            tmp.append((viseme_map[i][0], '000', viseme_map[i][1], viseme_map[i][2]))
        merge_set.append(tmp[0])
        words = merge_set[i][1]
        if(words is not None):
            words = words.lower()
            words_list.append(words)
    return merge_set


# In[46]:


# s = textual_features(transcript_set[0],viseme_set[0])
# s


# ## Own word2vec

# In[50]:


for i, elem in enumerate(transcript_set):
    s = textual_features(transcript_set[i],viseme_set[i])
    viseme_text_features.append(s)


# In[51]:


viseme_text_features


# In[ ]:


viseme_text_features


# In[ ]:


sentences = []
for filename in os.listdir(read_viseme_mod):
    file_path_read = os.path.join(read_viseme_mod, filename)
#     print(file_path_read)
    f_r = open(file_path_read,'r+')
    root = read_bml(f_r)
    script = read_transcript(root,0)
    sent = []
    for data in script :
        sent.append(data[0])
    sentences.append(sent)


# In[ ]:


model = Word2Vec(sentences, min_count=1,vector_size= 50,workers=3)                 
model.save("word2vec.model")       


'''
# In[47]:
# Loading word2vec directly

model = Word2Vec.load("word2vec.model")


# In[48]:


vocabulary = model.wv.key_to_index


    


# In[50]:


def create_mfcc(audio_path, start_ms, end_ms):
    start = start_ms/100
    end = end_ms/100
    y, sr = librosa.load(audio_path,sr=28000)
    y_cut = y[round(start*sr,ndigits=None)
         :round(end*sr, ndigits= None)]
    try:
        data = np.array([padding(librosa.feature.mfcc(y_cut,
             n_fft=n_fft,hop_length=hop_length,n_mfcc=128),128,50)])
    except:
        data = librosa.feature.mfcc(y_cut,
             n_fft=n_fft,hop_length=hop_length,n_mfcc=128)
        data = np.array([data[:, :50]])
    #data, sample_rate = lb.load(audio_path, offset = start, duration = end - start)
#     lb.display.waveplot(data)
#     print(type(data), type(sample_rate))
    #mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
#     librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
    #mfccScaled = np.mean(mfcc.T, axis=0)
    print(data.shape)
    return data

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    #print(h, w)
    a = (xx - h) // 2
    aa = xx - a - h
    b = (yy - w) // 2
    bb = yy - b - w
    print(h, w, xx, yy)
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

hop_length = 10 #the default spacing between frames
n_fft = 100


# In[53]:



#def feature_gneration(audio_gen):
features = []
labels = []
viseme_set = list()
transcript_set = list()
words_list = []
viseme_text_features = []

viseme_feature_set = list()

for filename in os.listdir(read_viseme_mod):
    file_path_read = os.path.join(read_viseme_mod, filename)
#     print(file_path_read)
    f_r = open(file_path_read,'r+', encoding = 'utf-8-sig')
    root = read_bml(f_r)
    viseme_extract = read_viseme(root)
    f_split = filename.split("_")
    audio_session = f_split[0][4]
    if(len(f_split)==4):
        audio_file_dir = f_split[0]+"_"+f_split[1]+"_"+f_split[2]
    else:
        audio_file_dir = f_split[0]+"_"+f_split[1]
    audio_file_name = filename.split(".")[0]+".wav"
    audio_path = audio_dir + "/Session" + audio_session + "/sentences/wav/"+audio_file_dir+"/"+audio_file_name
#     print(f_r)
    if(1):
        for rec in viseme_extract:
            #mfcc = create_mfcc(audio_path,float(rec[1]),float(rec[2])).tolist()
            #print(mfcc.shape, len(mfcc))
            #print(mfcc)
            #labels.append(rec[0])
            #features.append(mfcc)
            print("Done")
        #viseme_feature_set.append(mfcc_)
    viseme_set.append(viseme_extract)
#     print(viseme_extract[len(viseme_extract)-1][2])
    transcript_set.append(read_transcript(root,viseme_extract[len(viseme_extract)-1][2]) )
for i, elem in enumerate(transcript_set):
    s = textual_features(transcript_set[i],viseme_set[i])
    viseme_text_features.append(s)
print("This is viseme", len(viseme_text_features))
csv_file_text = 'test_textual_feature_set_new.csv'
fo = open(csv_file_text,'w')
csv_write = csv.writer(fo)
for sent in viseme_text_features:
    for tup in sent:
        word_vec = model.wv[tup[1]]
        
        word_vec = word_vec.tolist()
        word_vec.append(tup[0])
        csv_write.writerows([word_vec])
    csv_write.writerow([0]*51)
    


# In[77]:


csv_file_text = 'test_textual_feature_set_new_4.csv'
if os.path.exists(csv_file_text):
    os.remove(csv_file_text)
fo = open(csv_file_text,'a')
csv_write = csv.writer(fo)
write = 0
data_in = []
for sent in viseme_text_features:
    for tup in sent:
        write+= 1
        word_vec = model.wv[tup[1]]
        
        word_vec = word_vec.tolist()
        word_vec.append(tup[0])
        data_in.append(word_vec)
    data_in.append([0]*51)
    


# In[78]:


write


# In[79]:


len(data_in)


# In[80]:


text_features_set = pd.DataFrame(data_in)
text_features_set.to_csv(csv_file_text, index = False, header=False)


# In[66]:


visemes[:20]


# In[ ]:





# In[55]:


len(labels)


# In[62]:


len(viseme_text_features)


# In[57]:


viseme_text_features


# In[78]:


import pickle
pickle.dump(features, open('features_test.pkl', 'wb'))
pickle.dump(labels, open('labels_test.pkl', 'wb'))


# In[43]:


feature_gneration(1)


# In[24]:


get_ipython().system('dir')


# In[ ]:




