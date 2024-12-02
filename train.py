import os
import pdb
import platform
import time
from dataclasses import dataclass, field
from typing import Optional
from utils import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from GLM3.configuration_chatglm  import ChatGLMConfig
from GLM3.modeling_chatglm import ChatGLMForConditionalGeneration
from GLM3.tokenization_chatglm import ChatGLMTokenizer

TRAIN_FILES = [
    "/home/wangyu/data/baidubaike/baidubaike_563w_1.bin",
    "/home/wangyu/data/baidubaike/baidubaike_563w_2.bin",
    "/home/wangyu/data/baidubaike/baidubaike_563w_3.bin",
    "/home/wangyu/data/baidubaike/baidubaike_563w_4.bin",
    "/home/wangyu/data/baidubaike/baidubaike_563w_5.bin",
    "/home/wangyu/data/wiki/wiki.bin",
]

@dataclass
class PretrainArguments:
    model_save_dir: str = "./model_save/pre/"
    logs_dir: str = "./logs/"
    train_files: list = field(default_factory=lambda: TRAIN_FILES)
    max_seq_len: int = 512

pretrain_args = PretrainArguments()
config = ChatGLMConfig.from_pretrained("GLM3/config.json")
model = ChatGLMForConditionalGeneration(config,empty_init=False).bfloat16()
tokenizer = ChatGLMTokenizer.from_pretrained('GLM3')



train_dataset = PretrainDataset(pretrain_args.train_files, max_length=pretrain_args.max_seq_len, memmap=True)


model_size = sum(t.numel() for t in model.parameters())
print(f"chatglm3 size: {model_size / 1024 ** 2:.1f}M parameters")
def check_and_initialize_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Reinitializing {name} due to NaNs")
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)

check_and_initialize_parameters(model)

my_trainer_callback = MyTrainerCallback()

args = TrainingArguments(
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=64,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=500,
    ddp_find_unused_parameters=False,
    learning_rate=1e-3,
    save_steps=200,
    save_strategy="steps",
    save_total_limit=2,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    logging_steps=5,
    log_level="info",
    logging_first_step=True,
    bf16=True
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=train_dataset,
    callbacks=[my_trainer_callback],
)

# 添加训练前的参数统计信息


trainer.train(
   # resume_from_checkpoint='../../hy-tmp/model_save/pre/checkpoint-1693'
)


# eval_results = trainer.evaluate()
# print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
trainer.save_model(pretrain_args.model_save_dir)