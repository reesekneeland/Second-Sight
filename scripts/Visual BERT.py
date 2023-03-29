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
from torch import nn


# In[2]:


import torch
import clip
from PIL import Image


# In[3]:


model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[4]:


with open('COCO_images_captions.json') as f:
    data = json.load(f)


# In[5]:


data.keys()


# In[6]:


sub1file = pd.read_csv('./stim_list/stim_lists/CSI01_stim_lists.txt', sep='\n',header=None)


# In[7]:


text_sent = []
for i in sub1file[0]:
    if 'COCO_train' in i or 'rep_COCO_train' in i:
        i = i.replace('rep_','')
        text_sent.append(data[i][0])


# In[8]:


text_imagenet = open('./BOLD5000_Stimuli/Image_Labels/imagenet_final_labels.txt', 'r')
lines = text_imagenet.readlines()
text_imagenet_data = {}
for line in lines:
    if line.split(' ',1)[0].strip() not in text_imagenet_data:
        text_imagenet_data[line.split(' ',1)[0].strip()] = line.split(' ',1)[1].strip()


# In[ ]:


img_feat = np.load('../vilbert-multi-task/coco_frcnn.npy')
img_feat1 = np.load('../vilbert-multi-task/imagenet_bold_frcnn.npy')


# In[10]:


text_sent = []
img_feat2 = []
count = 0
count1 = 0
for i in sub1file[0]:
    i = i.replace('rep_','')
    if 'COCO_train' in i or 'rep_COCO_train' in i:
        text_sent.append(data[i][0])
        img_feat2.append(img_feat[count])
        #img_feat_boxes2.append(img_feat_boxes[count])
        count+=1
    elif 'n0' in i or ('n1' in i and 'n1.' not in i and 'n11.' not in i):
        #print(i.split('_')[0])
        text_sent.append(text_imagenet_data[i.split('_')[0]])
        img_feat2.append(img_feat1[count1])
        #img_feat_boxes2.append(img_feat_boxes1[count1])
        count1+=1
    else:
        text_sent.append(i.split('.')[0][:-1])
        img_feat2.append(img_feat1[count1])
        #img_feat_boxes2.append(img_feat_boxes1[count1])
        count1+=1


# In[11]:


img_feat2 = np.array(img_feat2)


# In[12]:


img_feat2.shape


# In[38]:


np.save('img_feat',np.reshape(img_feat,(img_feat.shape[0],img_feat.shape[2])))


# In[16]:


#language_output = []
#language_avg_output = []
vision_output = []
vision_avg_output = []
#pooled_output = []
#language_hidden_states = []
#vision_hidden_states = []
for i in np.arange(img_feat2.shape[0]):
    inputs = tokenizer(text_sent[i], return_tensors="pt",padding=True)
    visual_embeds = torch.Tensor(img_feat2[i].reshape(1,img_feat2[i].shape[0],img_feat2[i].shape[1]))
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
    vision_avg_output.append(nn.functional.adaptive_avg_pool2d(outputs['last_hidden_state'], (1,768)).detach().numpy())
    #pooled_output.append(outputs['pooled_output'].detach().numpy())
    #language_hidden_states.append(list(outputs['language_hidden_states']))
    #vision_hidden_states.append(list(outputs['vision_hidden_states']))


# In[17]:


vision_avg_output = np.array(vision_avg_output)
print(vision_avg_output.shape)


# In[19]:


np.save('visualbert_bold5000_avgpatch',vision_avg_output.reshape(vision_avg_output.shape[0],vision_avg_output.shape[3]))


# In[20]:


vision_output = np.array(vision_output)
print(vision_output.shape)


# In[21]:


np.save('visualbert_bold5000_pooled',vision_output.reshape(vision_output.shape[0],vision_output.shape[2]))


# In[13]:


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
for i in np.arange(img_feat2.shape[0]):
    inputs = tokenizer(text_sent[i], return_tensors="pt",padding=True)
    visual_embeds = torch.Tensor(img_feat2[i].reshape(1,img_feat2[i].shape[0],img_feat2[i].shape[1]))
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


# In[14]:


img_avg1 = np.array(vision_hidden2_output)
print(img_avg1.shape)


# In[15]:


img_avg3 = np.array(vision_hidden4_output)
print(img_avg3.shape)


# In[16]:


img_avg5 = np.array(vision_hidden6_output)
print(img_avg5.shape)


# In[17]:


img_avg7 = np.array(vision_hidden8_output)
print(img_avg7.shape)


# In[18]:


img_avg10 = np.array(vision_hidden9_output)
print(img_avg10.shape)


# In[19]:


img_avg12 = np.array(vision_hidden11_output)
print(img_avg12.shape)


# In[20]:


np.save('visualbert_img_feat_bold_layer2',np.reshape(img_avg1,(img_avg1.shape[0],img_avg1.shape[3])))
np.save('visualbert_img_feat_bold_layer4',np.reshape(img_avg3,(img_avg3.shape[0],img_avg3.shape[3])))
np.save('visualbert_img_feat_bold_layer6',np.reshape(img_avg5,(img_avg5.shape[0],img_avg5.shape[3])))
np.save('visualbert_img_feat_bold_layer8',np.reshape(img_avg7,(img_avg7.shape[0],img_avg7.shape[3])))
np.save('visualbert_img_feat_bold_layer9',np.reshape(img_avg10,(img_avg10.shape[0],img_avg10.shape[3])))
np.save('visualbert_img_feat_bold_layer11',np.reshape(img_avg12,(img_avg12.shape[0],img_avg12.shape[3])))


# In[ ]:




