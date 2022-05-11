import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report



test_labels = pickle.load(open('labels_test.pkl','rb'))
len(test_labels)



replace_dict_text = {'Ah':0, 'Ih':1, 'Z':2, 'W':3, 'D':4}




new_test_labels  = []
removes_indices = []
for ind,i in enumerate(test_labels):
    if i in replace_dict_text:
        new_test_labels.append(replace_dict_text[i])
    else:
        removes_indices.append(ind)



text_result = pd.read_csv('late_fusion_text_results.csv')



text_result['max_pred_proba'] = text_result[[f'{i}' for i in range(5)]].max(axis = 1).astype('float')
text_result['max_pred_class'] = text_result[[f'{i}' for i in range(5)]].idxmax(axis = 1).astype('float')

def late_fusion(y_pred_1, y_pred_1_prob, y_pred_2, y_pred_2_prob, dev_feat_label, data_name = 'test'):
    final_predictions = []
    for i in range(len(y_pred_1)):
        if y_pred_1[i] == y_pred_2[i]:
            final_predictions.append(y_pred_1[i])
        elif y_pred_1_prob[i] > y_pred_2_prob[i]:
            final_predictions.append(y_pred_1[i])
        else:
            final_predictions.append(y_pred_2[i])
    final_predictions = np.array(final_predictions)
    print("These are the metrics for ", data_name)
    print("F1-score ", f1_score(dev_feat_label, final_predictions, average = 'weighted'))
    print("Accuracy ", accuracy_score(dev_feat_label, final_predictions))
    print("Confusion Matrix ", confusion_matrix(dev_feat_label, final_predictions))
    return confusion_matrix(dev_feat_label, final_predictions)



saved_model = 'model_RNN_LSTM.h5'




import tensorflow as tf
model =  tf.keras.models.load_model(saved_model)





import pickle
data = pickle.load(open('features_test.pkl', 'rb'))



output=np.concatenate(data,axis=0)



o = np.delete(output, removes_indices, axis = 0)
o.shape



model_predictions = model.predict(o)


model_pred_class = np.argmax(model_predictions, axis=1) 
model_pred_proba = np.max(model_predictions, axis=1) 



late_fusion_cf = late_fusion(model_pred_class, model_pred_proba, text_result['max_pred_class'], text_result['max_pred_proba'], new_test_labels, data_name = 'test')



def calculate_acc_per_class(confusion_matrix):
    classes_ = []
    f1_score = []
    for class_ in range(5):
        classes_k = {}
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                if (i == j) and (i==class_):
                    classes_k['tp'] = confusion_matrix[i][j]
                elif (i == class_):
                    fp += confusion_matrix[i][j]
                elif (j == class_):
                    fn += confusion_matrix[i][j]
                else:
                    tn += confusion_matrix[i][j]
        classes_k['tn'] = tn
        classes_k['fp'] = fp
        classes_k['fn'] = fn
        classes_k['precision'] = classes_k['tp']/(classes_k['tp'] + fp)
        classes_k['recall'] = classes_k['tp']/(classes_k['tp'] + fn)
        classes_k['f1_score'] =(2* classes_k['precision']  * classes_k['recall']) /(classes_k['precision']  + classes_k['recall'] )
        f1_score.append(classes_k['f1_score'])
        classes_.append(classes_k) 
    f1_score_macro = sum(f1_score)/len(f1_score)
    return classes_, f1_score_macro
        
        
        
classes_, f1_macro = calculate_acc_per_class(late_fusion_cf)                  
                            

import seaborn as sns
import matplotlib.pyplot as pltNew


DetaFrame_cm = pd.DataFrame(late_fusion_cf, range(5), range(5))
sns.heatmap(DetaFrame_cm, annot=True)
pltNew.show()




