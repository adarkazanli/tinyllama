# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import time
import torch
from transformers import pipeline, LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer

config = LlamaConfig.from_pretrained("./TinyLlama-1.1B-Chat-v1.0/config.json")
localModel = AutoModelForCausalLM.from_pretrained("./TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("./TinyLlama-1.1B-Chat-v1.0/")

pipe = pipeline(task="text-generation",
                model=localModel,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                tokenizer=tokenizer)

# We use the tokenizers chat template to format each message - see
# https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a very sweet grandma chatbot",
    },
    {"role": "user", "content": "How to bake a cake"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
startTime=time.time()
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
endTime = time.time()
print(outputs[0]["generated_text"])
print(f"Time taken: {endTime-startTime} seconds")

# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...
