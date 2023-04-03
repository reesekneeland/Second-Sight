#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd
import os
from transformers import BertTokenizer, VisualBertModel
from transformers import ViTFeatureExtractor, ViTModel
import requests
import torch.nn as nn
import torch


# In[2]:


import torch
import clip
from PIL import Image


# In[3]:


model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[4]:


img_feat = np.load('../vilbert-multi-task/periera_img1.npy')
img_feat2 = np.load('../vilbert-multi-task/periera_img2.npy')
img_feat = np.concatenate([img_feat,img_feat2], axis=0)
print(img_feat.shape)


# In[5]:


import json
  
# Opening JSON file
f = open('concept2caption.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)


# In[6]:


text_sent = []
for eachword in sorted(data['concept2caption'].keys()):
    #print(eachword)
    if len(data['concept2caption'][eachword])!=6:
        print(len(data['concept2caption'][eachword]))
        print(eachword)
    for eachsent in data['concept2caption'][eachword]:
        #print(eachsent)
        text_sent.append(eachsent)


# In[7]:


remove_indices = [264, 379, 558, 610, 674, 675, 692, 758, 780, 782, 866, 897, 1005, 1009, 1013]


# In[8]:


img_feat = np.delete(img_feat, remove_indices, axis=0)
print(img_feat.shape)


# In[39]:


#language_output = []
#language_avg_output = []
vision_output = []
vision_avg_output = []
#pooled_output = []
#language_hidden_states = []
#vision_hidden_states = []
for i in np.arange(img_feat.shape[0]):
    inputs = tokenizer(text_sent[i], return_tensors="pt",padding=True)
    visual_embeds = torch.Tensor(img_feat[i].reshape(1,img_feat[i].shape[0],img_feat[i].shape[1]))
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    inputs.update({
     "visual_embeds": visual_embeds,
     "visual_token_type_ids": visual_token_type_ids,
     "visual_attention_mask": visual_attention_mask
     })
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    #language_output.append(outputs['language_output'].detach().numpy())
    #language_avg_output.append(np.mean(outputs['language_output'].detach().numpy(),axis=1))
    vision_output.append(outputs['pooler_output'].detach().numpy())
    vision_avg_output.append(np.mean(outputs['last_hidden_state'].detach().numpy(), axis=1))
    #pooled_output.append(outputs['pooled_output'].detach().numpy())
    #language_hidden_states.append(list(outputs['language_hidden_states']))
    #vision_hidden_states.append(list(outputs['vision_hidden_states']))


# In[43]:


vision_embeddings = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_embeddings.append(np.mean(vision_output[i:i+lt],axis=0))
    i+=lt


# In[44]:


vision_embeddings = np.array(vision_embeddings)
print(vision_embeddings.shape)


# In[46]:


vision_avg_embeddings = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_avg_embeddings.append(np.mean(vision_avg_output[i:i+lt],axis=0))
    i+=lt


# In[47]:


vision_avg_embeddings = np.array(vision_avg_embeddings)
print(vision_avg_embeddings.shape)


# In[26]:


vision_avg_output = np.array(vision_avg_output)
print(vision_avg_output.shape)


# In[27]:


np.save('visualbert_coco_vision_pool',vision_avg_output.reshape(vision_avg_output.shape[0],vision_avg_output.shape[2]))


# In[28]:


vision_output = np.array(vision_output)
print(vision_output.shape)


# In[30]:


np.save('visualbert_coco_vision_avg',vision_output.reshape(vision_output.shape[0],vision_output.shape[2]))


# In[45]:


np.save('visualbert_periera_lastlayer',vision_embeddings.reshape(vision_embeddings.shape[0],vision_embeddings.shape[2]))


# In[48]:


np.save('visualbert_periera_lastlayer_avgpatch',vision_avg_embeddings.reshape(vision_avg_embeddings.shape[0],vision_avg_embeddings.shape[2]))


# In[10]:


#language_output = []
#language_avg_output = []
vision_hidden1_output = []
vision_hidden2_output = []
vision_hidden3_output = []
vision_hidden4_output = []
vision_hidden5_output = []
vision_hidden6_output = []
vision_hidden7_output = []
vision_hidden8_output = []
vision_hidden9_output = []
vision_hidden10_output = []
vision_hidden11_output = []
vision_hidden12_output = []
#vision_avg_output = []
#pooled_output = []
#language_hidden_states = []
#vision_hidden_states = []
for i in np.arange(img_feat.shape[0]):
    inputs = tokenizer(text_sent[i], return_tensors="pt",padding=True)
    visual_embeds = torch.Tensor(img_feat[i].reshape(1,img_feat[i].shape[0],img_feat[i].shape[1]))
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    inputs.update({
     "visual_embeds": visual_embeds,
     "visual_token_type_ids": visual_token_type_ids,
     "visual_attention_mask": visual_attention_mask
     })
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    #language_output.append(outputs['language_output'].detach().numpy())
    #language_avg_output.append(np.mean(outputs['language_output'].detach().numpy(),axis=1))
    #vision_output.append(outputs['pooler_output'].detach().numpy())
    #vision_avg_output.append(np.mean(outputs['last_hidden_state'].detach().numpy(), axis=1))
    #pooled_output.append(outputs['pooled_output'].detach().numpy())
    #language_hidden_states.append(list(outputs['language_hidden_states']))
    #vision_hidden_states.append(list(outputs['vision_hidden_states']))
    vision_hidden2_output.append(nn.functional.adaptive_avg_pool2d(outputs['hidden_states'][2], (1,768)).detach().numpy())
    vision_hidden4_output.append(nn.functional.adaptive_avg_pool2d(outputs['hidden_states'][4], (1,768)).detach().numpy())
    vision_hidden6_output.append(nn.functional.adaptive_avg_pool2d(outputs['hidden_states'][6], (1,768)).detach().numpy())
    vision_hidden8_output.append(nn.functional.adaptive_avg_pool2d(outputs['hidden_states'][8], (1,768)).detach().numpy())
    vision_hidden9_output.append(nn.functional.adaptive_avg_pool2d(outputs['hidden_states'][9], (1,768)).detach().numpy())
    vision_hidden11_output.append(nn.functional.adaptive_avg_pool2d(outputs['hidden_states'][11], (1,768)).detach().numpy())


# In[11]:


img_avg1 = np.array(vision_hidden2_output)
img_avg1 = np.reshape(img_avg1, (img_avg1.shape[0], img_avg1.shape[3]))
vision_embeddings1 = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_embeddings1.append(np.mean(img_avg1[i:i+lt],axis=0))
    i+=lt
vision_embeddings1 = np.array(vision_embeddings1)
print(vision_embeddings1.shape)


# In[12]:


img_avg3 = np.array(vision_hidden4_output)
img_avg3 = np.reshape(img_avg3, (img_avg3.shape[0], img_avg3.shape[3]))
vision_embeddings3 = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_embeddings3.append(np.mean(img_avg3[i:i+lt],axis=0))
    i+=lt
vision_embeddings3 = np.array(vision_embeddings3)
print(vision_embeddings3.shape)


# In[13]:


img_avg5 = np.array(vision_hidden6_output)
img_avg5 = np.reshape(img_avg5, (img_avg5.shape[0], img_avg5.shape[3]))
vision_embeddings5 = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_embeddings5.append(np.mean(img_avg5[i:i+lt],axis=0))
    i+=lt
vision_embeddings5 = np.array(vision_embeddings5)
print(vision_embeddings5.shape)


# In[14]:


img_avg7 = np.array(vision_hidden8_output)
img_avg7 = np.reshape(img_avg7, (img_avg7.shape[0], img_avg7.shape[3]))
vision_embeddings7 = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_embeddings7.append(np.mean(img_avg7[i:i+lt],axis=0))
    i+=lt
vision_embeddings7 = np.array(vision_embeddings7)
print(vision_embeddings7.shape)


# In[15]:


img_avg10 = np.array(vision_hidden9_output)
img_avg10 = np.reshape(img_avg10, (img_avg10.shape[0], img_avg10.shape[3]))
vision_embeddings10 = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_embeddings10.append(np.mean(img_avg10[i:i+lt],axis=0))
    i+=lt
vision_embeddings10 = np.array(vision_embeddings10)
print(vision_embeddings10.shape)


# In[16]:


img_avg12 = np.array(vision_hidden11_output)
img_avg12 = np.reshape(img_avg12, (img_avg12.shape[0], img_avg12.shape[3]))
vision_embeddings12 = []
i = 0
for eachword in sorted(data['concept2caption'].keys()):
    lt = len(data['concept2caption'][eachword])
    vision_embeddings12.append(np.mean(img_avg12[i:i+lt],axis=0))
    i+=lt
vision_embeddings12 = np.array(vision_embeddings12)
print(vision_embeddings12.shape)


# In[17]:


np.save('visualbert_img_layer2_periera',vision_embeddings1)
np.save('visualbert_img_layer4_periera',vision_embeddings3)
np.save('visualbert_img_layer6_periera',vision_embeddings5)
np.save('visualbert_img_layer8_periera',vision_embeddings7)
np.save('visualbert_img_layer9_periera',vision_embeddings10)
np.save('visualbert_img_layer11_periera',vision_embeddings12)


# In[ ]:




