import os
import json
from typing import Dict

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def create_prompt(message: Dict[str, str]) -> str:
    """
    Creates a formatted prompt for training based on a message.
    We use the Llama 3 chat template to teach the model to generate
    responses in a specific style.
    """
    system_message = "You are an assistant that mimics a real person's communication style. Your task is to generate a message that is as similar as possible to this person's style."
    user_prompt = "Write a typical message in this person's style."
    assistant_response = message['content']

    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_response}<|eot_id|>"


def run_training(speaker: str):
    """
    Runs the model fine-tuning process for the specified speaker.
    """
    print(f"Starting training for speaker: {speaker}")

    data_path = f"./data/{speaker}_messages.json"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = [{"text": create_prompt(msg)} for msg in data["messages"]]
    dataset = Dataset.from_list(formatted_data)

    print(f"Dataset created with {len(dataset)} examples.")

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    output_dir = f"./output_lora/{speaker}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=1000,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting fine-tuning process...")
    trainer.train()
    print("Fine-tuning completed.")

    trainer.save_model(output_dir)
    print(f"LoRA adapter saved to {output_dir}")