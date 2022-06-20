import torch
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer, BartTokenizer, T5Tokenizer, RobertaTokenizer
from transformers import BertTokenizerFast, BartTokenizerFast, T5TokenizerFast, RobertaTokenizerFast

class VOX_CustomDataset(Dataset):
    def __init__(self, src_list, trg_list, max_len, tokenizer):

        # Assertion
        assert len(src_list) == len(trg_list)

        # Tokenizer Setting
        if tokenizer == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        elif tokenizer == 'bart':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        elif tokenizer == 'T5':
            tokenizer = BartTokenizer.from_pretrained('T5-base')
        elif tokenizer == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.tensor_list = []
        for i in range(len(src_list)):
            # Source tensor
            out = tokenizer(src_list[i],
                            max_length=max_len,
                            padding='max_length',
                            truncation=True)
            input_ids = torch.tensor(out['input_ids'], dtype=torch.long)
            att_mask = torch.tensor(out['attention_mask'], dtype=torch.long)
            # Target tensor
            trg_tensor = torch.tensor(trg_list[i], dtype=torch.long)

            self.tensor_list.append((input_ids, att_mask, trg_list[i]))

        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return self.num_data

class CNNDM_CustomDataset(Dataset):
    def __init__(self, src_list, max_len, tokenizer):

        # Tokenizer Setting
        if tokenizer == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        elif tokenizer == 'bart':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        elif tokenizer == 'T5':
            tokenizer = BartTokenizer.from_pretrained('T5-base')
        elif tokenizer == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.tensor_list = []
        for i in range(len(src_list)):
            # Source tensor
            out = tokenizer(src_list[i],
                            max_length=max_len,
                            padding='max_length',
                            truncation=True)
            input_ids = torch.tensor(out['input_ids'], dtype=torch.long)
            att_mask = torch.tensor(out['attention_mask'], dtype=torch.long)

            self.tensor_list.append((input_ids, att_mask))

        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return self.num_data