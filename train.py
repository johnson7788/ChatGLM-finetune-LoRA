import os
import time
import tqdm
import json
import argparse
import torch
import numpy as np
import loralib as lora
from lora_utils.insert_lora import get_lora_model

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup

def parse_args():
    """
    返回arg变量和help
    :return:
    """
    parser = argparse.ArgumentParser(description="加载模型",formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-id', '--model_id', type=str, default='finetune_test', help='实验的名称和保存模型名称')
    parser.add_argument('-lr', "--lora_rank", type=int, default=32, help="lora的秩")
    parser.add_argument('-e', "--epoch", type=int, default=1, help="训练的epoch数量")
    parser.add_argument('-m', "--max_length", type=int, default=512, help="训练的数据的最大长度512")
    return parser.parse_args(), parser.print_help

args, helpmsg = parse_args()
checkpoint = "THUDM/chatglm-6b"
model_id = args.model_id
lora_rank = args.lora_rank
mixed_precision = 'bf16'
lora_config = {
    'r': lora_rank,   #矩阵的秩
    'lora_alpha':32,   # lora的超参数alpha
    'lora_dropout':0.05,  # lora的dropout
    'enable_lora':[True, False, True],  # 对应着q,k,v，在哪些地方使用lora
}

LR = 1e-4
BATCH = 1
MAX_LENGTH = args.max_length
NUM_EPOCHS = args.epoch
accumulate_step = 8
warm_up_ratio = 0.1

# 使用accelerate的deepseed插件
deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, deepspeed_plugin=deepspeed_plugin, log_with="tensorboard", project_dir='runs/')
device = accelerator.device  # accelerator自动选择设备

# 只在主进程上执行，加载模型和tokenizer
with accelerator.main_process_first():
    retry_cnt = 10
    cnt = 0
    while cnt < retry_cnt:
        try:
            import dataset.GLM
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')
            model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision = 'main')
            if mixed_precision == None:
                model = model.float()
            break
        except:
            cnt += 1 

    model = get_lora_model(model, lora_config)

# 等待每个workers都准备好
accelerator.wait_for_everyone()

model.use_cache = False
model.gradient_checkpointing = False


import dataset.beauty as Load_My_Dataset
dataset.GLM.device = device

accelerator.print('开始准备数据集')

with accelerator.main_process_first():
    # pairs = Load_My_Dataset.load('./data/alpaca_data.json')
    pairs = Load_My_Dataset.load('./data/问答数据集.json')
    pairs_encoded = dataset.GLM.encode_pairs(pairs, tokenizer)
    accelerator.print('未按最大长度过滤前的数据集大小：', len(pairs_encoded))
    pairs_encoded = list(filter(lambda pair: len(pair['prompt'])+len(pair['completion']) <= MAX_LENGTH, pairs_encoded))
    accelerator.print('按最大长度过滤后的数据集大小：', len(pairs_encoded))
train_dataset = dataset.GLM.SimpleDataset(pairs_encoded)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn = dataset.GLM.collate_fn, shuffle=True, batch_size=BATCH)


# 等待每个workers都准备好
accelerator.wait_for_everyone()



optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(len(train_dataloader) // accumulate_step * NUM_EPOCHS),
)
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)


# 日志名称记录
accelerator.init_trackers(model_id, {})

total_effective_step = 0

for epoch in range(NUM_EPOCHS):

    batch_loss = 0
    effective_step = 0
    
    for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):
        # batch批次数据来自collate_fn函数返回，返回input_ids,attention_mask,labels,position_ids
        outputs = model(**batch)

        loss_d = outputs.loss.detach().cpu().float().item()
        batch_loss += loss_d

        loss = outputs.loss / accumulate_step
        accelerator.backward(loss)

        if (step+1) % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            effective_step += 1

            gathered_batch_loss = accelerator.gather((torch.tensor(batch_loss, device=device)))

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "train_loss": gathered_batch_loss.mean().item() / accumulate_step,
                        "epoch": epoch,
                    },
                    step = total_effective_step + effective_step,
                )

            t.set_description(f"loss: {gathered_batch_loss.mean().item() / accumulate_step}")
            batch_loss = 0   
        
    
    accelerator.wait_for_everyone()
    
    total_effective_step += effective_step
    
    if accelerator.is_main_process:
        os.makedirs(f'saved/{model_id}', exist_ok = True)
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), f'saved/{model_id}/{model_id}_epoch_{epoch}.pt')

    accelerator.wait_for_everyone()

