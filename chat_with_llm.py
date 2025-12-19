import  os
from collections import deque
from openai import OpenAI

def interact_with_llm(prompt, model="deepseek-v3"):

    
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="your key",  ## set your key to use LLM api service; current service is from aliyuncs
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
                    model=model,  
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ]
                )
    content = completion.choices[0].message.content

    return content
