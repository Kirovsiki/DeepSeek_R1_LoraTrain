import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import re

# ================= 配置区 =================
BASE_MODEL_PATH = "/sg-platform-deepseek/DeepSeek-R1-Distill-Qwen-32B"
LORA_ADAPTER_PATH = "./fine_tuned_model"
SYSTEM_PROMPT = """#角色：你是"""
EOS_TOKEN = "</s>"  
GENERATION_CONFIG = {
    "temperature": 0.3,  
    "top_p": 0.7,
    "top_k": 40,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.2,
    "do_sample": True,
    "pad_token_id": 0,  
}
# ==========================================

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_PATH,
        load_in_4bit = False,
        max_seq_length = 2048,
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    print("加载LoRA适配器...")
    try:
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        print("LoRA适配器成功加载！")
    except Exception as e:
        print("LoRA适配器加载失败！可能的原因包括：")
        print("1. LoRA适配器路径不正确。")
        print("2. LoRA适配器文件损坏或格式不正确。")
        print("3. 基础模型与LoRA适配器不兼容。")
        print(f"错误详情: {str(e)}")
        raise RuntimeError("无法加载LoRA适配器，请检查上述原因并重试。")

    model = model.merge_and_unload()
    model = FastLanguageModel.for_inference(model)
    
    # 设置EOS token ID
    global GENERATION_CONFIG
    GENERATION_CONFIG["eos_token_id"] = tokenizer.eos_token_id

    return model, tokenizer

def build_prompt(user_input):
    return f"""role:"system","content":"#角色：你是"
"role:"user","content":{user_input}
"role:"assistant","content":
<think>"""

def extract_conclusion(text):
    full_pattern = re.compile(r'<think>(.*?)</think>\s*结论：(.*?)(?:</s>|$)', re.DOTALL)
    full_match = full_pattern.search(text)
    
    if full_match:
        thinking = full_match.group(1).strip()
        conclusion = full_match.group(2).strip()
        return thinking, conclusion
    think_pattern = re.compile(r'<think>(.*?)(?:</think>|$)', re.DOTALL)
    think_match = think_pattern.search(text)
    thinking = think_match.group(1).strip() if think_match else ""
    conclusion_pattern = re.compile(r'结论：(.*?)(?:</s>|$)', re.DOTALL)
    conclusion_match = conclusion_pattern.search(text)
    conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
    if not conclusion:
        cleaned_text = re.sub(r'</?think>|</s>|<\|.*?\|>|结论：', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        conclusion = cleaned_text.strip()
    
    return thinking, conclusion

def generate_response(model, tokenizer, prompt, show_thinking=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    inputs = tokenizer(
        [prompt],
        return_tensors = "pt",
        padding = True,
    ).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **GENERATION_CONFIG,
        )
    full_response = tokenizer.decode(output[0], skip_special_tokens=False)
    response_text = full_response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)):]
    thinking, conclusion = extract_conclusion(response_text)
    if not thinking or not conclusion:
        pass
    
    return thinking, conclusion

def reset_model(full_reload=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if full_reload:
        global model, tokenizer
        model, tokenizer = load_model()
        model.eval()
        return True
    
    return False

def interactive_chat():
    show_thinking = False
    reset_level = "light"  
    
    while True:
        try:
            user_input = input("\n您：").strip()
            if user_input.lower() in ("exit", "quit"):
                print("退出对话。")
                break
            if user_input.lower() == "toggle":
                show_thinking = not show_thinking
                print(f"{'显示' if show_thinking else '隐藏'}思考过程。")
                continue
            if user_input.lower() == "reload":
                print("正在完全重新加载模型...")
                reset_model(full_reload=True)
                print("模型已完全重新加载。")
                continue
            if user_input.lower() == "reset":
                reset_level = "light" if reset_level == "full" else "full"
                print(f"重置级别已切换为: {'完全重载' if reset_level == 'full' else '轻量级'}")
                continue
            if not user_input:
                print("输入不能为空！")
                continue
            prompt = build_prompt(user_input)
            
            print("助手：", end="", flush=True)
            thinking, conclusion = generate_response(model, tokenizer, prompt)
            if show_thinking and thinking:
                print("\n思考过程：")
                print(thinking)
                print("\n结论：")
            if conclusion:
                print(conclusion)
            else:
                print("抱歉，我无法生成有效的回答。请尝试重新提问或重载模型。")
            
            print("="*60)
            if reset_level == "full":
                print("[系统正在完全重新加载模型...]")
                reset_model(full_reload=True)
                print("[模型已重新加载，准备接收新问题]")
            else:
                print("[系统已轻量级重置，准备接收新问题]")
                reset_model(full_reload=False)

        except KeyboardInterrupt:
            print("\n对话终止。")
            break
        except Exception as e:
            print(f"\n发生错误：{str(e)}")
            reset_model(full_reload=True)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model, tokenizer = load_model()
    model.eval() 

    with torch.inference_mode():
        interactive_chat()
