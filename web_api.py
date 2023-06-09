#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import logging
import tqdm
import peft
import torch
import numpy as np
import loralib as lora
from peft import LoraConfig
from itertools import compress
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
# 日志保存的路径，保存到当前目录下的logs文件夹中
log_path = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)
logfile = os.path.join(log_path, "api.log")
# 日志的格式
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(logfile, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class QKV_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(QKV_layer, self).__init__()
        self.linear_q = torch.nn.Linear(in_features, out_features // 3)
        self.linear_k = torch.nn.Linear(in_features, out_features // 3)
        self.linear_v = torch.nn.Linear(in_features, out_features // 3)

    def update(self, target_layer):
        self.linear_q.weight.data = target_layer.weight[:target_layer.out_features // 3, :].data
        self.linear_q.bias.data = target_layer.bias[:target_layer.out_features // 3].data

        self.linear_k.weight.data = target_layer.weight[
                                    target_layer.out_features // 3:target_layer.out_features // 3 * 2, :].data
        self.linear_k.bias.data = target_layer.bias[
                                  target_layer.out_features // 3:target_layer.out_features // 3 * 2].data

        self.linear_v.weight.data = target_layer.weight[target_layer.out_features // 3 * 2:, :].data
        self.linear_v.bias.data = target_layer.bias[target_layer.out_features // 3 * 2:].data

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return torch.concat([q, k, v], dim=-1)


def get_lora_model(model, lora_config, update=False):
    # q,k,v中的哪些位置使用lora； update表示是否更新lora的参数
    target_modules = list(compress(['q', 'k', 'v'], lora_config['enable_lora']))
    # 加载peft的lora配置
    config = LoraConfig(
        peft_type="LORA",
        task_type="CAUSAL_LM",
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=target_modules,
        lora_dropout=lora_config['lora_dropout'])

    pbar = tqdm.tqdm(total=28, desc="更新成lora模型")
    # 遍历模型的所有层的名称和模块
    for key, module in model.named_modules():
        if key.endswith('attention'):
            layer = int(key.split('.')[2])
            if isinstance(module.query_key_value, peft.tuners.lora.LoraModel):
                if update:
                    qkv_layer = QKV_layer(module.query_key_value.model.linear_q.in_features,
                                          module.query_key_value.model.linear_q.in_features * 3)
                    qkv_layer.linear_q.load_state_dict(module.query_key_value.model.linear_q.state_dict(), strict=False)
                    qkv_layer.linear_k.load_state_dict(module.query_key_value.model.linear_k.state_dict(), strict=False)
                    qkv_layer.linear_v.load_state_dict(module.query_key_value.model.linear_v.state_dict(), strict=False)
                    module.query_key_value = qkv_layer
                    module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)
                else:
                    continue
            else:
                qkv_layer = QKV_layer(module.query_key_value.in_features, module.query_key_value.out_features)
                qkv_layer.update(module.query_key_value)
                module.query_key_value = qkv_layer  # 更新模块
                module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)

            pbar.update(1)

    pbar.close()

    lora.mark_only_lora_as_trainable(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in model_parameters])

    model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    non_trainable_params = sum([np.prod(p.size()) for p in model_parameters])

    print('trainable_params:{} ({:.2f}%), non_trainable_params:{}'.format(trainable_params,
                                                                          trainable_params / non_trainable_params * 100,
                                                                          non_trainable_params))

    return model

def get_response(text):
    """
    暂时没用到
    """
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

@app.route("/api/chat", methods=['POST'])
def chat():
    """
    Args: 基于aspect的情感分析，给定实体，判断实体在句子中的情感
    """
    jsonres = request.get_json()
    # 可以预测多条数据
    data = jsonres.get('data', None)
    if not data:
        return jsonify({"code": 400, "msg": "data不能为空"}), 400
    logging.info(f"数据分别是: {data}")
    input = data.get('text', '')
    history = data.get('history', [])
    response, history = model.chat(tokenizer, input, history=history)
    result = {"response": response}
    logging.info(f"返回的结果是: {result}")
    return jsonify(result)

@app.route("/ping", methods=['GET', 'POST'])
def ping():
    """
    测试
    :return:
    :rtype:
    """
    return jsonify("Pong")

def parse_args():
    """
    返回arg变量和help
    :return:
    """
    parser = argparse.ArgumentParser(description="加载模型",formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--device', type=str, default="cuda", choices=("cuda","cpu"), help='使用的设备')
    parser.add_argument('-m', '--model_name_or_path', type=str, default='THUDM/chatglm-6b', help='使用的base模型')
    parser.add_argument('-e', "--enable_lora", action='store_true', help="是否开启lora模型")
    parser.add_argument('-lr', "--lora_rank", type=int, default=32, help="lora的秩")
    parser.add_argument('-l', '--lora_path', type=str, default="saved/finetune_test/finetune_test_epoch_0.pt",  help='使用的lora模型')
    return parser.parse_args(), parser.print_help

def load_model(args):
    device = args.device
    checkpoint = args.model_name_or_path
    lora_model = args.lora_path
    enable_lora = args.enable_lora
    lora_rank = args.lora_rank
    assert os.path.exists(lora_model), f"lora模型不存在: {lora_model}"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision='main')
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision='main')
    if enable_lora:
        lora_config = {
            'r': lora_rank,
            'lora_alpha': 32,
            'lora_dropout': 0.5,
            'enable_lora': [True, False, True],
        }
        model = get_lora_model(model, lora_config)
        _ = model.load_state_dict(torch.load(lora_model), strict=False)
    if device == "cuda":
        model.half().cuda().eval()
    else:
        model.cpu().eval()
    return model, tokenizer

if __name__ == '__main__':
    args, helpmsg = parse_args()
    model, tokenizer = load_model(args)
    app.run(host='0.0.0.0', port=7087, debug=False, threaded=True)

