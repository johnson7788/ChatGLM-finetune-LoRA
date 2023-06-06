#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["https_proxy"] = 'http://127.0.0.1:7890'
#os.environ["http_proxy"] = 'http://127.0.0.1:7890'

import torch
from transformers import AutoTokenizer, AutoModel

device = 'cuda'
checkpoint = "THUDM/chatglm-6b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')


# In[2]:


import loralib as lora
from lora_utils.insert_lora import get_lora_model


# In[3]:


lora_config = {
        'r': 8,
        'lora_alpha':16,
        'lora_dropout':0.1,
        'enable_lora':[True, False, True],
    }


# In[4]:


model = get_lora_model(model, lora_config)


# In[5]:


_ = model.load_state_dict(torch.load('saved/chatglm-6b_demo.pt'), strict=False)


# In[6]:


_ = model.half().cuda().eval()


# In[7]:



role = '峰哥'

question = f'{role}能锐评一下大语言模型吗？'

emotional = '真诚的'
length = '详细的'

text=f'{question}\n{role}{emotional}{length}答：'


# In[9]:


inputs = tokenizer(text, padding=True, truncation=True, max_length=1024, return_tensors="pt")
inputs = {k: v.cuda() for k,v in inputs.items() }
outputs = model.generate(
    **inputs, 
    max_length=1024,
    eos_token_id=130005,
    do_sample=True,
    temperature=0.75,
    top_p = 0.75,
    top_k = 10000,
    repetition_penalty=1.5, 
    num_return_sequences=10,
)


# In[11]:


for output in outputs:
    print(tokenizer.decode(output)[len(text):])


# In[ ]:




