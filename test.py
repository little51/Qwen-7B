from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig

model_path = "./dataroot/models/Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:0",
    quantization_config=quantization_config,
    trust_remote_code=True,
).eval()
model.generation_config = GenerationConfig.from_pretrained(
    model_path, trust_remote_code=True)
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
response, history = model.chat(
    tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
