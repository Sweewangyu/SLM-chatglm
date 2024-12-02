import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState
class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length, memmap=True):
        super().__init__()
        if memmap:
            with open(data_path_lst[0], 'rb') as f:
                data = np.fromfile(f, dtype=np.uint16, count=max_length * 10)

            # Use os.path.getsize() to get the actual file size in bytes
            file_size_bytes = os.path.getsize(data_path_lst[0])
            flen = file_size_bytes // np.dtype('uint16').itemsize
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





class MyTrainerCallback(TrainerCallback):
    log_cnt = 0
    loss_values = []  # 用于存储损失值

    def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs=None,
            **kwargs,
    ):
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

        # 记录损失值
        if logs and "loss" in logs:
            self.loss_values.append(logs["loss"])

    def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        control.should_save = True
        return control

    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        # 在训练结束时绘制损失曲线
        self.plot_loss_curve()

    def plot_loss_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_values, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig("loss_curve.png")


