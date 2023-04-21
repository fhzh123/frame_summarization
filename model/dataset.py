import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer, src_list: list = list(), trg_list: list = None, 
                 min_len: int = 10, src_max_len: int = 768, trg_max_len: int = 300):

        self.tokenizer = tokenizer
        self.src_tensor_list = list()
        self.trg_tensor_list = list()

        self.min_len = min_len
        self.src_max_len = src_max_len

        for src in src_list:
            if min_len <= len(src):
                self.src_tensor_list.append(src)

        for trg in trg_list:
            if min_len <= len(trg):
                self.trg_tensor_list.append(trg)

        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):

        src_encoded_dict = \
        self.tokenizer(
            self.src_tensor_list[index],
            max_length=self.src_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src_input_ids = src_encoded_dict['input_ids'].squeeze(0)
        src_attention_mask = src_encoded_dict['attention_mask'].squeeze(0)

        trg_encoded_dict = \
        self.tokenizer(
            self.trg_tensor_list[index],
            max_length=self.trg_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        trg_input_ids = trg_encoded_dict['input_ids'].squeeze(0)
        trg_attention_mask = trg_encoded_dict['attention_mask'].squeeze(0)

        return (src_input_ids, src_attention_mask), (trg_input_ids, trg_attention_mask)

    def __len__(self):
        return self.num_data