import json


PROMPT_DICT = {
    "prompt_input": (
        "下面是描述任务的一个指令, 输入和提供的上下文是配对的。 "
        "根据提问写一个合适的回答\n\n"
        "### 指令:\n{instruction}\n\n### 上下文:\n{input}\n\n### 回答:"
    ),
    "prompt_no_input": (
        "下面是描述任务的一个指令。"
        "根据提问写一个合适的回答。\n\n"
        "### 指令:\n{instruction}\n\n### 回答:"
    ),
}

def load(path):

    with open(path, 'r') as f:
        content = json.load(f)

    pairs = []

    for line in content:
        if line['input'] == '':
            prompt = PROMPT_DICT['prompt_no_input'].format_map(line)
        else:
            prompt = PROMPT_DICT['prompt_input'].format_map(line)
        completion = line['output']
        pairs.append({'prompt':prompt, 'completion':completion})
    
    return pairs