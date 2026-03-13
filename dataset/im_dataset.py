# _*_ coding : utf-8 _*_
# @Time : 2026/3/10 17:01
# @Author : phy013x
# @File : im_dataset

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    # init
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.tokenizer = tokenizer

        # 输入给GPU的最大长度
        self.max_length = max_length

        # 使用HuggingFace datasets的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path)

    # __len__
    def __len__(self):
        return len(self.samples)

    # __getitem__
    # 我们拿到的数据是，jsonl的每一行
    def __getitem__(self, index):
        sample = self.samples[index]

        # tokenizer把文本转化为input_id
        tokens = self.tokenizer(
            str(sample["text"]).__init__(), # 这里假设jsonl里有一个"text"字段，包含了文本内容
            add_special_tokens=False,
            max_length=self.max_length - 2, # 留出位置给BOS和EOS
            truncation=True, # 如果长度超过max，自动剪切
        ).input_ids

        # 需要加上EOS，BOS，以及PAD填充
        tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens)) # 填充到max_length
        input_ids = torch.Tensor(input_ids, dtype=torch.long)

        # 需要自行编写labels，防止PAD参与loss计算
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100 # 将PAD的label设为-100，表示忽略这些位置的loss计算

        # 需要编写attention_mask，告诉模型哪些位置是有效的，哪些位置是PAD
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long() # 非PAD设置为1，PAD设置为0

        # 我们要输出的是，input_ids, attention_mask, labels
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


















