#!/usr/bin/env python
# coding: utf-8

import os

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["https_proxy"] = 'http://127.0.0.1:7890'
#os.environ["http_proxy"] = 'http://127.0.0.1:7890'

import torch
from transformers import AutoTokenizer, AutoModel

device = 'cuda'
checkpoint = "THUDM/chatglm-6b"
lora_model = "saved/finetune_test/finetune_test_epoch_0.pt"
# lora_model = "saved/chatglm-6b_demo.pt"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')


import loralib as lora
from lora_utils.insert_lora import get_lora_model


lora_config = {
        'r': 32,
        'lora_alpha':32,
        'lora_dropout':0.5,
        'enable_lora':[True, False, True],
    }

# lora_config = {
#         'r': 8,
#         'lora_alpha':16,
#         'lora_dropout':0.1,
#         'enable_lora':[True, False, True],
#     }


# model = get_lora_model(model, lora_config)
# _ = model.load_state_dict(torch.load(lora_model), strict=False)

_ = model.half().cuda().eval()

# role = '峰哥'

# question = f'{role}夏季去海边度假有哪些必备的护肤品？'
# question = f'夏季去海边度假有哪些必备的护肤品？'
    question = f'水杨酸到底是什么，有什么用？该怎么用？'

# emotional = '真诚的'
# length = '详细的'

# text=f'{question}\n{role}{emotional}{length}答：'
text = f'{question}'


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
    num_return_sequences=1,  #生成几个句子
)

for output in outputs:
    print(tokenizer.decode(output)[len(text):])



