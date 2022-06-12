import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, src_list, src_att_list, trg_list, trg_att_list,
                 src_max_len: int = 300, trg_max_len: int = 360,
                 pad_idx: int = 0, eos_idx: int = 2):
        # Stop index list
        stop_ix_list = [pad_idx, eos_idx]
        self.tensor_list = []
        for src, src_att, trg, trg_att in zip(src_list, src_att_list, trg_list, trg_att_list):
            if src[src_max_len-1] in stop_ix_list and trg[trg_max_len-1] in stop_ix_list:
                # Source tensor
                src_tensor = torch.tensor(src[:src_max_len], dtype=torch.long)
                src_att_tensor = torch.tensor(src_att[:src_max_len], dtype=torch.long)
                # Target tensor
                trg_tensor = torch.tensor(trg[:trg_max_len], dtype=torch.long)
                trg_att_tensor = torch.tensor(trg_att[:trg_max_len], dtype=torch.long)
                # tensor list
                self.tensor_list.append((src_tensor, src_att_tensor, trg_tensor, trg_att_tensor))

        self.num_data = len(self.tensor_list)

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return self.num_data