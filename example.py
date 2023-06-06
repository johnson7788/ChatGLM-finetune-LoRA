#!/usr/bin/env python
# coding: utf-8

# ## Load Model From huggingface

# In[4]:


import os

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["https_proxy"] = 'http://127.0.0.1:7890'
#os.environ["http_proxy"] = 'http://127.0.0.1:7890'

from transformers import AutoTokenizer, AutoModel

device = 'cuda'
checkpoint = "THUDM/chatglm-6b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')


# ## Insert LoRA to model

# In[5]:


import loralib as lora
from lora_utils.insert_lora import get_lora_model


# In[6]:


lora_config = {
        'r': 8,
        'lora_alpha':16,
        'lora_dropout':0.1,
        'enable_lora':[True, False, True],
    }


# In[7]:


model = get_lora_model(model, lora_config)


# ## Dataset

# In[8]:


device = 'cuda'


# In[9]:


import dataset.GLM 
from torch.utils.data import DataLoader

dataset.GLM.device = device
#dataset.GLM.pad_to = 8


# In[10]:


pairs = [{'prompt':'你好', 'completion':'你好, 我是ChatGLM'}]
pairs_encoded = dataset.GLM.encode_pairs(pairs, tokenizer)
train_dataset = dataset.GLM.SimpleDataset(pairs_encoded)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn = dataset.GLM.collate_fn, shuffle=True, batch_size=1)


# ## Training

# In[11]:


model.half().to(device)


# In[12]:


batch = {k: v.to(device) for k, v in next(iter(train_dataloader)).items()}


# In[14]:


model(**batch).loss


# ## Inference

# In[19]:


import torch


# In[86]:


pairs = [
    {'prompt':'周末适合哪里玩?', 'completion':'周末适合去上海'},
    {'prompt':'周末适合哪里玩?', 'completion':'周末适合去北京'},
]

pairs_encoded = dataset.GLM.encode_pairs(pairs, tokenizer, with_eos=False)
test_dataset = dataset.GLM.SimpleDataset(pairs_encoded)
test_dataloader = DataLoader(dataset=test_dataset, collate_fn = dataset.GLM.collate_fn, shuffle=True, batch_size=1)


# In[87]:


batch = {k: v.to(device) for k, v in next(iter(test_dataloader)).items()}


# In[88]:


outputs = model.generate(
    **batch, 
    max_length=1024,
    eos_token_id=130005,
    do_sample=True,
    temperature=0.55,
    top_p = 0.75,
    top_k = 10000,
    repetition_penalty=1.5, 
    num_return_sequences=1,

    )


# In[89]:


for output in outputs:
    print(tokenizer.sp_tokenizer.decode(output))


# ## Chat

# In[92]:


response, history = model.chat(tokenizer, "如何缓解焦虑", history=[])


# In[93]:


response


# ## Load pretrain weight

# In[97]:


model.load_state_dict(torch.load('saved/chatglm-6b_alpaca_5.pt'), strict=False)


# In[ ]:




