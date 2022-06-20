# Import modules
import os
import gc
import psutil
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score
# Import custom modules
from task.frame.dataset import CNNDM_CustomDataset
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name

from transformers import BertForSequenceClassification, RobertaForSequenceClassification, T5Model, BartForSequenceClassification

def cnn_frame_detection(args):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    # Data Load
    dat = pd.read_csv(os.path.join(args.data_path, 'cnn_dailymail', args.cnn_dailymail_ver, 'train.csv'))
    dat = dat.dropna()

    gc.enable()
    write_log(logger, "Finished loading data!")

    # 2) Dataloader setting
    custom_dataset = CNNDM_CustomDataset(dat['summary'].tolist(), max_len=args.src_max_len, tokenizer=args.tokenizer)
    custom_dataloader = DataLoader(custom_dataset, drop_last=False, batch_size=args.batch_size,
                                   shuffle=False, pin_memory=True, num_workers=args.num_workers)
    write_log(logger, f"Total number of trainingsets  iterations - {len(custom_dataset)}, {len(custom_dataloader)}")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    if args.model_type == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=5)
    elif args.model_type == 'bart':
        model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=5)
    elif args.model_type == 'T5':
        model = T5Model.from_pretrained('facebook/bart-base', num_labels=5)
    elif args.model_type == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)

    save_path = os.path.join(args.model_save_path, args.tokenizer)
    save_file_name = os.path.join(save_path, 
                                  f'checkpoint_v_{args.variational_mode}_p_{args.parallel}.pth.tar')
    checkpoint = torch.load(save_file_name)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'start!')

    softmax = nn.Softmax(dim=1)
    frame_list = list()

    with torch.no_grad():
        for i, batch_iter in enumerate(tqdm(custom_dataloader, bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):
                
            # Input, output setting
            src_sequence = batch_iter[0]
            src_att = batch_iter[1]

            src_sequence = src_sequence.to(device, non_blocking=True)
            src_att = src_att.to(device, non_blocking=True)

            predicted = model(input_ids=src_sequence, attention_mask=src_att)
            out = predicted.logits

            softmax_out = softmax(out)

            for prob in softmax_out:
                if max(prob) >= 0.85:
                    frame_list.append(torch.argmax(prob).item())
                else:
                    frame_list.append(99)

    dat['frame'] = frame_list
    dat.to_csv('cnn_frame_dat.csv', index=False)