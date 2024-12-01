import os
import pdb
import platform
import time
from dataclasses import dataclass, field
from typing import Optional

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

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

TRAIN_FILES = [
    # "/home/wangyu/data/baidubaike/baidubaike_563w_1.bin",
    # "/home/wangyu/data/baidubaike/baidubaike_563w_2.bin",
    # "/home/wangyu/data/baidubaike/baidubaike_563w_3.bin",
    # "/home/wangyu/data/baidubaike/baidubaike_563w_4.bin",
    # "/home/wangyu/data/baidubaike/baidubaike_563w_5.bin",
    "/home/wangyu/data/wiki/wiki.bin",
]

@dataclass
class PretrainArguments:
    tokenizer_dir: str = "./GLM3/"
    model_save_dir: str = "./model_save/pre/"
    logs_dir: str = "./logs/"
    train_files: list = field(default_factory=lambda: TRAIN_FILES)
    max_seq_len: int = 512

pretrain_args = PretrainArguments()
tokenizer = ChatGLMTokenizer.from_pretrained(pretrain_args.tokenizer_dir)


class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length=pretrain_args.max_seq_len, memmap=True):
        super().__init__()
        if memmap:
            with open(data_path_lst[0], 'rb') as f:
                data = np.fromfile(f, dtype=np.uint16, count=max_length * 10)
                print("First few records from the file:", data)

            # Use os.path.getsize() to get the actual file size in bytes
            file_size_bytes = os.path.getsize(data_path_lst[0])
            flen = file_size_bytes // np.dtype('uint16').itemsize
            print(f"Total number of uint16 elements in file: {flen}")

            # Load using memmap
            self.data = np.memmap(data_path_lst[0], dtype=np.uint16, shape=(flen // max_length, max_length))
        else:
            data_lst = []
            for data_path in data_path_lst:
                with open(data_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)
                    data_lst.append(data)
            data = np.concatenate(data_lst)
            data = data[:max_length * int(len(data) / max_length)]
            self.data = data.reshape(-1, max_length)

        # 打印数据集的大小和类型
        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape))
        print("Data type:", self.data.dtype)
        print("downloading finished.....")

        # 验证数据集
        if len(self.data) == 0:
            raise ValueError("Data is empty after loading")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return {'input_ids': torch.from_numpy(X), 'labels': torch.from_numpy(Y)}


train_dataset = PretrainDataset(pretrain_args.train_files, max_length=pretrain_args.max_seq_len, memmap=True)
config = ChatGLMConfig.from_pretrained("GLM3/config.json")


model = ChatGLMForConditionalGeneration(config, empty_init=False).bfloat16()
model_size = sum(t.numel() for t in model.parameters())
print(f"chatglm3 size: {model_size / 1024 ** 2:.1f}M parameters")
# 检查并重新初始化模型参数
def check_and_initialize_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Reinitializing {name} due to NaNs")
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)

check_and_initialize_parameters(model)
model_size = sum(t.numel() for t in model.parameters())
print(f"chatglm3 size: {model_size / 1024 ** 2:.1f}M parameters")
def print_model_stats(model):
    for name, param in model.named_parameters():
        print(f"{name}: mean={param.mean().item()}, std={param.std().item()}")

print("Model stats before training:")
print_model_stats(model)
class MyTrainerCallback(TrainerCallback):
    log_cnt = 0
    def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        control.should_save = True
        return control

my_trainer_callback = MyTrainerCallback()

args = TrainingArguments(
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,
    num_train_epochs=1,
    weight_decay=0.1,
    ddp_find_unused_parameters=False,
    learning_rate=5e-4,
    save_steps=500,
    save_strategy="steps",
    save_total_limit=2,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    logging_steps=20,
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

# 添加训练后的参数统计信息
# print("Model stats after training:")
# print_model_stats(model)

# eval_results = trainer.evaluate()
# print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
# trainer.save_model(pretrain_args.model_save_dir)
# import os
# os.system('shutdown')