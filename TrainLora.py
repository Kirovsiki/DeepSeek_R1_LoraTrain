import os
import torch
torch.cuda.empty_cache()
import time
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from tqdm import tqdm
import argparse

# 设置环境变量
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1' 
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '10'  

# 全局变量
loss_history = []

class LoggingCallback(TrainerCallback):
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.progress_bar.update(1)
            loss = logs.get("loss", 0)
            self.progress_bar.set_postfix(loss=loss)
            loss_history.append(loss)
            print(f"Step: {state.global_step}, Loss: {loss}")

def load_model_with_retry(model_name, max_retries=5, retry_delay=60, **kwargs):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to load model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                **kwargs
            )
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception("Failed to load model after multiple attempts.")
###拉去或者是自己定义/自己下载更好
def main(args):
    print("Loading model and tokenizer...")
    # 修改后的本地模型路径（新定义）
    model, tokenizer = load_model_with_retry(
        model_name="/home/kud/main/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
        max_retries=5,
        retry_delay=5,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # model, tokenizer = load_model_with_retry(
    #     model_name="unsloth/DeepSeek-R1-Distill-Qwen-14B",
    #     max_retries=5,
    #     retry_delay=5,
    #     max_seq_length=args.max_seq_length,
    #     dtype=None,
    #     load_in_4bit=True,
    # )


    print("Preparing PEFT model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("Loading and processing dataset...")
    ###这里是整理system prompt，需要你自己定义模型的过程
    train_prompt_style = """systemPromptxxx

    ### Question:
    {}

    ### Response:
    <think>
    {}
    </think>
    {}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        inputs = examples["Question"]
        cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for input, cot, output in zip(inputs, cots, outputs):
            text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[0:500]", trust_remote_code=True)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("Setting up trainer...")
    training_args = TrainingArguments(
        per_device_train_batch_size=2,  # 2
        gradient_accumulation_steps=4,  # 4
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        disable_tqdm=True,  
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        args=training_args,
    )

    print("Starting training...")
    progress_bar = tqdm(total=training_args.max_steps, desc="Training")
    logging_callback = LoggingCallback(progress_bar)
    trainer.add_callback(logging_callback)

    trainer_stats = trainer.train()
    progress_bar.close()

    print("Training complete. Saving model...")
    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")

    print("Training process complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    args = parser.parse_args()

    main(args)