import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from peft import PeftModel
from threading import Thread

# ================= 配置区 =================
BASE_MODEL_PATH = "/home/kud/main/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit"
LORA_ADAPTER_PATH = "/home/kud/main/DeepSeek/outputs/checkpoint-60"
SYSTEM_PROMPT = """您是一位拥有临床推理、诊断和治疗方案制定专业知识的医疗专家。请以专业且准确的方式回答以下医学问题。"""

GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.2,
    "do_sample": True,
}
# ==========================================

def load_model():
    """加载并优化模型"""
    print("正在加载基础模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_PATH,
        load_in_4bit = True,
        max_seq_length = 2048,
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    model = model.merge_and_unload()
    
    # 新增关键步骤：准备推理优化
    model = FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def build_prompt(user_input):
    """构建符合训练格式的提示"""
    return f"""<|beginoftext|>system
{SYSTEM_PROMPT}<|endoftext|>
<|startofuser|>用户
{user_input}<|endoftext|>
<|startofassistant|>助手
"""

def streaming_generation(model, tokenizer, prompt):
    """流式生成响应"""
    inputs = tokenizer(
        [prompt],
        return_tensors = "pt",
        padding = True,
    ).to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        **GENERATION_CONFIG,
        **inputs,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("助手：", end="", flush=True)
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(new_text, end="", flush=True)
    print("\n" + "="*60)
    return generated_text

def interactive_chat():
    """交互式对话循环"""
    print("医疗助手已就绪（输入 'exit' 退出）...")
    while True:
        try:
            user_input = input("\n您：").strip()
            if user_input.lower() in ("exit", "quit"):
                print("退出对话。")
                break
            if not user_input:
                print("输入不能为空！")
                continue

            prompt = build_prompt(user_input)
            _ = streaming_generation(model, tokenizer, prompt)

        except KeyboardInterrupt:
            print("\n对话终止。")
            break
        except Exception as e:
            print(f"\n发生错误：{str(e)}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model, tokenizer = load_model()
    model.eval()  # 确保在评估模式
    
    # 新增：验证模型是否准备好
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    
    with torch.inference_mode():
        interactive_chat()